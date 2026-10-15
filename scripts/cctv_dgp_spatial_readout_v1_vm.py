"""Human-manual L4 reconstruction comparison; verify/export are not qualification."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cctv_dgp_spatial_readout_v1_contract import (
    NAME, STEM, PARENT_NAME, BUDGETS, read, write, sha, require_manual,
    verify_transfer, common_radius, project_original_ball, stage_gate,
)

STAGE = 'outputs/spatial_readout_v1'


def machine_guard(root, idle=False):
    require_manual(root)
    request = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/machine-type',
                                     headers={'Metadata-Flavor': 'Google'})
    with urllib.request.urlopen(request, timeout=3) as response:
        assert response.read().decode().rstrip().endswith('/g2-standard-4')
    if idle:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                capture_output=True, text=True, check=True, timeout=10)
        assert not result.stdout.strip(), 'Competing GPU workload; preserve it'


def load_dependencies(root, parent, p):
    # This function is reachable only after machine_guard, never local verification.
    machine_guard(root)
    sys.path.insert(1, str(parent))
    import numpy as np
    import torch
    from cctv_dgp_bank_comparison_v1_model import ConditionedBankDGP
    from cctv_dgp_bank_pinned_initializer_v2 import load_pinned_initializer
    from cctv_dgp_generative_bank_v1 import load_prior
    from cctv_dgp_pilot import FixedObservedIdentity, state_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    assert torch.__version__ == '2.9.1+cu129', 'Retain environment mismatch; do not install automatically'
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == 'NVIDIA L4'
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261014)
    torch.cuda.manual_seed_all(20261014)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    original, provenance = load_frozen_dgp_restorer(parent/'weights/dgp_v2.pth',
        expected_sha256=p['original_checkpoint_sha256'], device='cuda')
    identity = FixedObservedIdentity(parent/'weights/w600k_r50.onnx', 'cuda')
    prior = load_prior(parent/'prior_source', parent/'bank_implementation', p['standalone_prior_sha256']).cuda()
    provider = ConditionedBankDGP(original.net, prior,
        torch.from_numpy(np.load(parent/'initializers/mean_style.npy', allow_pickle=False)).cuda()).cuda().eval().requires_grad_(False)
    loading = dict(torch=torch.__version__, loaded_on='cuda', learning_enabled=False)
    load_pinned_initializer(provider, parent/'initializers/initial_B.pth',
                            p['parent_required_sha256']['initializers/initial_B.pth'], 'B', loading)
    assert state_hash(original.net) == p['original_state'] and state_hash(identity) == p['recognizer_state']
    assert state_hash(provider) == p['feature_provider_initial_state']
    return np, torch, original, identity, provider, provenance, loading


def train(root, parent, p, old, pin):
    machine_guard(root, idle=True)
    out = root/STAGE
    assert not (root/'outputs').exists(), 'Used immutable pilot; no unchanged rerun'
    out.mkdir(parents=True)
    start = time.monotonic()
    counts = dict(optimizer_updates=0, gradient_queries=0, backwards=0, original_batch_calls=0,
                  prior_feature_batch_calls=0, decoder_batch_calls=0, recognizer_batch_calls=0)
    progress = dict(stage='pre-model resource check', route=None, route_updates=0)
    candidate = optimizer = provider = original = identity = None
    np = torch = None
    radius = None
    finished_routes = []

    def clock():
        assert time.monotonic()-start < p['budgets']['worker_seconds'], 'Worker finite time stop'
        assert shutil.disk_usage(root).free >= p['budgets']['disk_reserve_bytes'], 'Disk reserve stop'
        assert sum(path.stat().st_size for path in out.rglob('*') if path.is_file()) <= p['budgets']['output_byte_cap'], 'Output cap stop'
        if torch is not None:
            torch.cuda.synchronize()
            assert torch.cuda.max_memory_allocated() <= p['budgets']['peak_VRAM_bytes'], 'VRAM stop'
            import resource
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= p['budgets']['peak_RSS_bytes'], 'RAM stop'
        for key, value in counts.items():
            maximum = p['budgets'].get('maximum_'+key, p['budgets'].get(key))
            assert value <= maximum, key+' finite count stop'

    def stop(_signal, _frame):
        raise RuntimeError('Finite supervisor stop signal; retain partial evidence')
    signal.signal(signal.SIGTERM, stop)

    def save_state(label):
        if candidate is None:
            return
        folder = out/('state_'+label)
        folder.mkdir()
        torch.save(dict(model=candidate.state_dict(), optimizer=None if optimizer is None else optimizer.state_dict(),
            CPU_rng=torch.get_rng_state(), CUDA_rng=torch.cuda.get_rng_state_all(), route=progress['route'],
            route_updates=progress['route_updates'], protocol_sha256=pin, counts=dict(counts),
            failed_recipe_resume_allowed=False, exact_future_resume_guaranteed=False), folder/'full_state.pth')
        from cctv_dgp_pilot import state_hash, buffer_hash
        write(folder/'receipt.json', dict(model_state=state_hash(candidate), buffer_state=buffer_hash(candidate),
            full_state_sha256=sha(folder/'full_state.pth'), route=progress['route'], updates=progress['route_updates'],
            optimizer_present=optimizer is not None, model_qualified=False))

    try:
        free = shutil.disk_usage(root).free
        mem = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
        write(out/'storage_start.json', dict(free_bytes=free, MemAvailable_bytes=mem['MemAvailable'], neural_calls=0, optimizer_updates=0))
        assert free >= p['budgets']['minimum_free_bytes'], 'Need8GiB free after installation; retain stop'
        assert mem['MemAvailable'] >= p['budgets']['available_RAM_before_import_bytes'], 'Need8GiB available RAM before imports'
        np, torch, original, identity, provider, provenance, loading = load_dependencies(root, parent, p)
        from PIL import Image
        from cctv_dgp_pilot import grid112, state_hash, buffer_hash
        from cctv_dgp_head4_lossless_v1 import unpack
        from cctv_dgp_group_conflicts_v1_losses import pixel_and_structure_losses
        from cctv_dgp_bank_photometric_guard_v1_loss import brightness_guard
        from frozen_capacity_contract import feature_support, groups, exported_pixel_metrics, detail_metric
        from frozen_raw_metrics import deliver, mean_only, pixel_metrics, detail_float
        from cctv_dgp_spatial_readout_v1_model import build_model
        torch.cuda.reset_peak_memory_stats()
        write(out/'environment.json', dict(host=os.uname().nodename, GPU=torch.cuda.get_device_name(0),
            python=sys.version, CUDA=torch.version.cuda, packages={n: importlib.metadata.version(n) for n in
            ['torch', 'torchvision', 'numpy', 'Pillow', 'scipy', 'scikit-image', 'onnx', 'onnx2torch']},
            DGP_provenance=provenance, initializer=loading, AMP=False, TF32=False,
            original_state=state_hash(original.net), recognizer_state=state_hash(identity), provider_state=state_hash(provider)))
        refs = {r['id']: r for r in old['references']}
        case_map = {c['id']: c for c in old['cases']}
        caches = []
        def pixels(path, mode='RGB'):
            with Image.open(path) as image:
                assert image.size == (256, 256)
                return np.asarray(image.convert(mode)).copy()
        def tensor(arrays):
            # Preserve app NumPy division before CUDA transfer, including PNG replay.
            return torch.from_numpy(np.stack(arrays).astype(np.float32)/np.float32(255)).permute(0, 3, 1, 2).contiguous().cuda()
        def array_hash(array):
            import hashlib
            return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        def emb(image, b):
            counts['recognizer_batch_calls'] += 1
            return identity.embedding(image, b['mask'], b['grid'])
        def gpu(b):
            return {key: value.cuda() for key, value in b['CPU'].items()} | dict(
                maps=[value.cuda() for value in b['maps']], features={k: v.cuda() for k, v in b['features'].items()})

        stamp = time.monotonic()
        cache_rows = []
        for serial, block in enumerate(p['reference_blocks']):
            progress['stage'] = 'frozen feature cache'
            cs = [case_map[c] for c in block['case_ids']]
            ref = refs[block['reference']]
            cameras = [pixels(parent/c['input']) for c in cs]
            targets = [pixels(parent/ref['target']) for c in cs]
            masks = [pixels(parent/ref['observed'], 'L') > 0 for _ in cs]
            support = [feature_support(mask, c['landmarks5_canvas_xy']) for mask, c in zip(masks, cs)]
            matrices = [grid112(ref['matrix112']) for _ in cs]
            b = dict(cases=cs, camera=cameras, target8=targets, mask8=masks, support8=support)
            values = dict(x=tensor(cameras), target=tensor(targets),
                mask=torch.from_numpy(np.stack(masks).astype(np.float32))[:, None].cuda(),
                feature=torch.from_numpy(np.stack(support).astype(np.float32))[:, None].cuda(),
                grid=torch.from_numpy(np.stack(matrices)).cuda())
            with torch.no_grad():
                counts['original_batch_calls'] += 1
                baseline = torch.where(values['mask'].bool(), original(values['x']), values['x'])
                maps = provider.current.observe(values['x'])
                counts['prior_feature_batch_calls'] += 1
                features = provider.conditioned_features(maps)
                old_terms = pixel_and_structure_losses(baseline, values['target'], values['mask'], values['feature'])
            b['baseline'] = baseline.cpu().permute(0, 2, 3, 1).numpy().copy()
            retained_path = parent/'outputs/bank_comparison_v1/baseline/packs'/f"b{block['begin']//5:03d}.npz"
            retained, retained_embeddings = unpack(retained_path, block['case_ids'])
            assert np.array_equal(b['baseline'], retained), 'Fresh original RAW differs from retained baseline; no optimizer'
            b['retained_embeddings'] = retained_embeddings
            b['target_vectors'] = retained_embeddings[:, 2].copy()
            b['baseline_vectors'] = retained_embeddings[:, 0].copy()
            values['baseline'] = baseline
            values['target_vector'] = torch.from_numpy(b['target_vectors']).cuda()
            values['old_arc'] = 1-(torch.from_numpy(b['baseline_vectors']).cuda().double()*values['target_vector'].double()).sum(1)
            b['CPU'] = {k: v.detach().cpu().contiguous() for k, v in values.items()}
            b['old_terms'] = {k: v.detach().cpu() for k, v in old_terms.items()}
            b['maps'] = [v.detach().cpu().contiguous() for v in maps]
            b['features'] = {k: v.detach().cpu().contiguous() for k, v in features.items()}
            caches.append(b)
            cache_rows.append(dict(reference=block['reference'], maps=[array_hash(v.numpy()) for v in b['maps']],
                features={k: array_hash(v.numpy()) for k, v in b['features'].items()},
                source=block['source'], bytes=sum(v.numel()*v.element_size() for v in b['maps']+list(b['features'].values()))))
            del values, baseline, maps, features, old_terms, retained, retained_embeddings
            clock()
            elapsed = time.monotonic()-stamp
            if serial == 1:
                projected = elapsed/2*20*1.5
                write(out/'cache_projection.json', dict(references=2, seconds=elapsed, projected_seconds=projected, cap_seconds=240, factor=1.5))
                assert projected <= 240, 'Cache timing projection stop'
            assert elapsed <= 240, 'Cache finite time stop'
            print(json.dumps(dict(cache_references=serial+1, of=20)), flush=True)
        write(out/'cache_receipt.json', dict(rows=cache_rows, seconds=time.monotonic()-stamp,
            storage='ephemeral CPU float32 derived cache; hashes retained; no TRAIN assets removed', cap_seconds=240))
        torch.save(provider.state_dict(), out/'frozen_feature_provider.pth')

        def metric_rows(label, model=None, exact_parity=False):
            stamp = time.monotonic()
            folder = out/label
            folder.mkdir()
            (folder/'packs').mkdir()
            (folder/'png').mkdir()
            rows = []
            for index, b in enumerate(caches):
                g = gpu(b)
                with torch.no_grad():
                    if model is None:
                        pred = g['baseline']
                    else:
                        counts['decoder_batch_calls'] += 1
                        pred = torch.where(g['mask'].bool(), model.forward_cached(g['x'], g['maps'], g['features']), g['x'])
                    raw = pred.cpu().permute(0, 2, 3, 1).numpy().copy()
                    png = np.stack([deliver(a, camera, mask) for a, camera, mask in zip(raw, b['camera'], b['mask8'])])
                    if exact_parity:
                        assert np.array_equal(raw, b['baseline']), 'Initial100 RAW parity differs; no optimizer'
                        assert all(np.array_equal(a, deliver(z, c, m)) for a, z, c, m in zip(png, b['baseline'], b['camera'], b['mask8'])), 'Initial100 PNG parity differs'
                    zero_png = np.stack([deliver(z, c, m) for z, c, m in zip(b['baseline'], b['camera'], b['mask8'])])
                    counts['recognizer_batch_calls'] += 2
                    vr = identity.embedding(torch.cat([pred, g['baseline'], g['target']]),
                        torch.cat([g['mask']]*3), torch.cat([g['grid']]*3)).cpu().numpy()
                    vp = identity.embedding(torch.cat([tensor(list(png)), tensor(list(zero_png)), g['target']]),
                        torch.cat([g['mask']]*3), torch.cat([g['grid']]*3)).cpu().numpy()
                    for values, slot in [(vr, 0), (vp, 1)]:
                        np.testing.assert_allclose(values[5:10], b['retained_embeddings'][:, slot], rtol=1e-6, atol=1e-6)
                        np.testing.assert_allclose(values[10:15], b['target_vectors'], rtol=1e-6, atol=1e-6)
                np.savez_compressed(folder/'packs'/f'r{index:02d}.npz', ids=np.asarray([c['id'] for c in b['cases']]),
                    raw=raw, embeddings_raw=vr, embeddings_png=vp, target_embeddings=b['target_vectors'])
                for i, c in enumerate(b['cases']):
                    a, mask, target = raw[i], b['mask8'][i], b['target8'][i]
                    assert np.array_equal(a[~mask], b['camera'][i][~mask].astype(np.float32)/np.float32(255))
                    control, control_png, _ = mean_only(a, b['baseline'][i], b['camera'][i], mask)
                    actual = pixel_metrics(a, target, mask)
                    raw_vector = b['retained_embeddings'][i, 0] if model is None or exact_parity else vr[i]
                    png_vector = b['retained_embeddings'][i, 1] if model is None or exact_parity else vp[i]
                    raw_target = b['target_vectors'][i] if model is None or exact_parity else vr[10+i]
                    png_target = b['target_vectors'][i] if model is None or exact_parity else vp[10+i]
                    actual.update(ArcFace_observed_fixed=float(raw_vector@raw_target),
                        landmark_high_frequency_MSE=detail_float(a, target, b['support8'][i]),
                        constant_mean_shift_only_MSE=pixel_metrics(control, target, mask)['MSE'])
                    delivered = exported_pixel_metrics(png[i], target, mask)
                    delivered.update(ArcFace_observed_fixed=float(png_vector@png_target),
                        landmark_high_frequency_MSE=detail_metric(png[i], target, b['support8'][i]),
                        constant_mean_shift_only_MSE=exported_pixel_metrics(control_png, target, mask)['MSE'])
                    Image.fromarray(png[i]).save(folder/'png'/(c['id']+'.png'))
                    rows.append(dict(case=c['id'], source=c['source'], profile=c['profile'], raw=actual, png=delivered))
                del g, pred, raw, png, vr, vp, zero_png
                clock()
                assert time.monotonic()-stamp <= 300, 'Snapshot finite time stop'
            reductions = {stage: groups([dict(source=r['source'], profile=r['profile'], metrics=r[stage]) for r in rows]) for stage in ['raw', 'png']}
            result = dict(rows=rows, groups=reductions, seconds=time.monotonic()-stamp)
            write(folder/'metrics.json', result)
            return result

        progress['stage'] = 'original snapshot'
        baseline_metrics = metric_rows('baseline')
        models, initial_vectors, parameter_layouts, norms = {}, {}, {}, []
        def flat(model):
            return np.concatenate([v.detach().cpu().numpy().reshape(-1) for v in model.learning_parameters()]).astype(np.float32)
        def assign(model, vector):
            offset = 0
            with torch.no_grad():
                for value in model.learning_parameters():
                    count = value.numel()
                    value.copy_(torch.from_numpy(vector[offset:offset+count].reshape(tuple(value.shape))).cuda())
                    offset += count
            assert offset == len(vector) and np.array_equal(flat(model), vector)
        for route in p['selected_routes']:
            model = build_model(root, parent, provider, root/'readouts.npz', route)
            assert model.learning_names() == p['trainable_parameter_names'][route]
            models[route] = model
            initial_vectors[route] = flat(model)
            norms.append(float(np.linalg.norm(initial_vectors[route].astype(np.float64))))
            offset, layout = 0, []
            for name, value in zip(model.learning_names(), model.learning_parameters()):
                layout.append(dict(name=name, shape=list(value.shape), begin=offset, end=offset+value.numel()))
                offset += value.numel()
            parameter_layouts[route] = layout
            np.save(out/(route+'_initial_vector.npy'), initial_vectors[route], allow_pickle=False)
            write(out/(route+'_parameter_layout.json'), layout)
        historical = np.concatenate([v.detach().cpu().numpy().reshape(-1) for name, v in provider.named_parameters()
                                    if name.startswith(('style_conditioner.', 'spatial_conditioners.'))])
        norms.append(float(np.linalg.norm(historical.astype(np.float64))))
        radius = common_radius(norms)
        write(out/'distance_bound.json', dict(norms_A_B_historical_conditioning=norms, common_radius=radius,
            distance_fraction_maximum=.001, additional_B_parameters_do_not_inflate_allowance=True))

        def objective(pred, g, b):
            terms = pixel_and_structure_losses(pred, g['target'], g['mask'], g['feature'])
            arc = 1-(emb(pred, g).double()*g['target_vector'].double()).sum(1)
            old_terms = {k: v.cuda() for k, v in b['old_terms'].items()}
            raw = [terms['MSE'].mean(), terms['SSIM_loss'].mean(), arc.mean(), terms['landmark_structure'][1:].mean()]
            normalizers = p['normalizers']
            barrier = 5*(torch.relu(terms['MSE']-old_terms['MSE']-1e-12).mean()/normalizers[0]
                + torch.relu(terms['SSIM_loss']-old_terms['SSIM_loss']-1e-6).mean()/normalizers[1]
                + torch.relu(arc-g['old_arc']-1e-6).mean()/normalizers[2])
            anchors = terms['MSE'][0]/p['active_anchor_normalizers'][0]+terms['MSE'][1]/p['active_anchor_normalizers'][1]
            photo, detail = brightness_guard(pred, g['baseline'], g['target'], g['mask'], normalizers[0], 10., .2)
            value = sum(w*v/n for w, v, n in zip(p['reconstruction_weights'], raw, normalizers))+barrier+anchors+photo
            assert torch.isfinite(value)
            return value, dict(objective=float(value.detach()), raw_terms=[float(v.detach()) for v in raw],
                               barrier=float(barrier.detach()), anchors=float(anchors.detach()), **detail)

        preflight = []
        for route, model in models.items():
            progress.update(stage='initial parity and derivatives', route=route, route_updates=0)
            candidate, optimizer = model, None
            metric_rows(route+'0', model, exact_parity=True)
            save_state(route+'0')
            model.enable_learning()
            preflight_start = time.monotonic()
            sums = np.zeros(len(model.learning_names()), dtype=np.float64)
            times = []
            for index in p['schedule'][:2]:
                clock()
                b, g = caches[index], gpu(caches[index])
                t = time.monotonic()
                counts['decoder_batch_calls'] += 1
                pred = torch.where(g['mask'].bool(), model.forward_cached(g['x'], g['maps'], g['features']), g['x'])
                loss, terms = objective(pred, g, b)
                derivatives = torch.autograd.grad(loss, model.learning_parameters(), allow_unused=False)
                counts['gradient_queries'] += 1
                assert all(torch.isfinite(v).all() for v in derivatives)
                values = np.asarray([float(v.detach().double().norm()) for v in derivatives])
                sums += values
                torch.cuda.synchronize()
                times.append(time.monotonic()-t)
                preflight.append(dict(route=route, reference=p['reference_blocks'][index]['reference'],
                    names=model.learning_names(), norms=values.tolist(), terms=terms, seconds=times[-1]))
                del derivatives, pred, loss, g
            assert (sums > 0).all(), 'Every selected reconstruction tensor needs nonzero finite gradients before optimizer'
            projection = max(times)*50*1.75+50*.25
            write(out/('preflight_'+route+'.json'), dict(rows=[r for r in preflight if r['route'] == route],
                summed_gradient_norms=sums.tolist(), all_nonzero=True, fit_projected_seconds=projection,
                fit_cap_seconds=600, projection_factor=1.75, journal_allowance_per_step_seconds=.25, optimizer_updates=0))
            assert projection <= 600, 'Measured fit timing projection stop before optimizer'
            assert time.monotonic()-preflight_start <= 180, 'Arm preflight finite time stop'
        projected_total = time.monotonic()-start+sum(read(out/('preflight_'+r+'.json'))['fit_projected_seconds'] for r in models)+2*baseline_metrics['seconds']*1.5
        write(out/'worker_projection.json', dict(projected_seconds=projected_total, cap_seconds=3000, before_optimizer=True))
        assert projected_total <= 3000, 'Whole-worker timing projection stop before optimizer'
        # Strict pessimistic uncompressed journal bound: three float32 vectors
        # per step, plus baseline/initial/final arrays, checkpoints and reserves.
        projected_output = 3*50*4*sum(len(v) for v in initial_vectors.values())+5*100*256*256*3*4+400*1024**2
        write(out/'output_projection.json', dict(projected_uncompressed_bytes=projected_output,
            cap_bytes=p['budgets']['output_byte_cap'], export_reserve_bytes=projected_output+64*1024**2))
        assert projected_output <= p['budgets']['output_byte_cap']
        assert shutil.disk_usage(root).free >= 2*projected_output+p['budgets']['disk_reserve_bytes']+64*1024**2

        for route, model in models.items():
            candidate = model
            progress.update(stage='finite bounded fit', route=route, route_updates=0)
            selected = model.learning_parameters()
            # Adam updates shadow values. A proposal outside the scientific ball
            # is never installed in, or evaluated by, the live decoder.
            shadow = [torch.nn.Parameter(value.detach().clone(), requires_grad=False) for value in selected]
            optimizer = torch.optim.Adam(shadow, lr=.0001, betas=(.9, .999), eps=1e-8,
                weight_decay=1e-5, foreach=False, fused=False)
            frozen_before = state_hash({n: v for n, v in model.state_dict().items() if n not in model.learning_names()})
            buffers_before = buffer_hash(model)
            journal = out/('trajectory_'+route)
            journal.mkdir()
            rows = []
            stamp = time.monotonic()
            for update, index in enumerate(p['schedule'], 1):
                clock()
                assert time.monotonic()-stamp <= 600, 'Arm fit finite time stop'
                b, g = caches[index], gpu(caches[index])
                optimizer.zero_grad(set_to_none=True)
                model.zero_grad(set_to_none=True)
                counts['decoder_batch_calls'] += 1
                pred = torch.where(g['mask'].bool(), model.forward_cached(g['x'], g['maps'], g['features']), g['x'])
                loss, terms = objective(pred, g, b)
                loss.backward()
                counts['backwards'] += 1
                assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in selected)
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(selected, 1.).detach())
                clipped_gradient = np.concatenate([v.grad.detach().cpu().numpy().reshape(-1) for v in selected]).astype(np.float32)
                for proposal_value, model_value in zip(shadow, selected):
                    proposal_value.grad = model_value.grad.detach().clone()
                optimizer.step()
                counts['optimizer_updates'] += 1
                progress['route_optimizer_steps'] = update
                proposal = np.concatenate([v.detach().cpu().numpy().reshape(-1) for v in shadow]).astype(np.float32)
                projected, bound = project_original_ball(initial_vectors[route], proposal, radius)
                assign(model, projected)
                with torch.no_grad():
                    for proposal_value, model_value in zip(shadow, selected):
                        proposal_value.copy_(model_value)
                progress['route_updates'] = update
                np.savez_compressed(journal/f'update_{update:03d}.npz', clipped_gradient=clipped_gradient,
                                    unconstrained=proposal, actual=projected)
                row = dict(update=update, reference=p['reference_blocks'][index]['reference'], cache_index=index,
                    gradient_norm_before_clip=gradient_norm, projection=bound, **terms)
                rows.append(row)
                with (journal/'trace.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                assert buffer_hash(model) == buffers_before
                if update % 10 == 0:
                    assert state_hash({n: v for n, v in model.state_dict().items() if n not in model.learning_names()}) == frozen_before
                    assert state_hash(provider) == p['feature_provider_initial_state']
                    print(json.dumps(dict(route=route, update=update, of=50, objective=terms['objective'], distance=bound['actual_distance'])), flush=True)
                del g, pred, loss, proposal, projected, clipped_gradient
            write(out/('fit_'+route+'.json'), dict(rows=rows, updates=50, seconds=time.monotonic()-stamp,
                cap_seconds=600, sample_passes=2.5, full_corpus_epochs=0, frozen_state=frozen_before, buffer_state=buffers_before))
            progress['stage'] = 'actual RAW and PNG endpoint'
            save_state(route+'50')
            metrics = metric_rows(route+'50', model)
            gates = {stage: stage_gate(baseline_metrics['groups'][stage], metrics['groups'][stage]) for stage in ['raw', 'png']}
            outcome = dict(route=route, updates=50, gates=gates, numerical_sample_pass=all(g['pass'] for g in gates.values()),
                           development_model_accepted=False, app_promotion=False, independent_image_review_required=True)
            finished_routes.append(outcome)
            write(out/('outcome_'+route+'.json'), outcome)
            print(json.dumps(dict(route=route,
                RAW_structure_gain_percent=100*gates['raw']['global_structure_gain'],
                PNG_structure_gain_percent=100*gates['png']['global_structure_gain'],
                RAW_failed_checks=len(gates['raw']['preservation_failures']),
                PNG_failed_checks=len(gates['png']['preservation_failures']),
                numerical_sample_pass=outcome['numerical_sample_pass'], model_qualified=False)), flush=True)
            if not outcome['numerical_sample_pass']:
                write(out/('quality_stop_'+route+'.json'), dict(message='Early structure/preservation requirement failed; retain stop',
                    outcome=outcome, automatic_extra_updates=0, failed_recipe_resume_allowed=False))
            optimizer = None
        assert state_hash(original.net) == p['original_state'] and state_hash(identity) == p['recognizer_state']
        assert state_hash(provider) == p['feature_provider_initial_state']
        write(out/'run_results.json', dict(complete=True, outcomes=finished_routes, counts=counts,
            seconds=time.monotonic()-start, original_state_unchanged=True, recognizer_state_unchanged=True,
            feature_provider_state_unchanged=True, model_qualified=False, full_corpus_epochs=0,
            accepted_development_updates=0, automatic_extra_epochs=False))
        return 0 if all(r['numerical_sample_pass'] for r in finished_routes) else 2
    except BaseException as exc:
        failure = dict(UTC=datetime.now(timezone.utc).isoformat(), type=type(exc).__name__, message=str(exc),
            traceback=traceback.format_exc(), progress=progress, counts=counts, seconds=time.monotonic()-start,
            completed_routes=finished_routes, model_qualified=False, no_automatic_resume=True)
        try:
            if candidate is not None:
                save_state(str(progress['route'])+'_stopped')
        except BaseException as state_exc:
            failure['state_export_error'] = repr(state_exc)
        write(out/'failure.json', failure)
        raise


def replay(root, parent, p, old):
    machine_guard(root, idle=True)
    out = root/STAGE
    assert read(out/'run_results.json')['complete']
    np, torch, original, identity, provider, _, _ = load_dependencies(root, parent, p)
    from PIL import Image
    from cctv_dgp_pilot import grid112, state_hash
    from cctv_dgp_spatial_readout_v1_model import build_model
    from frozen_raw_metrics import deliver
    refs = {r['id']: r for r in old['references']}
    cases = {c['id']: c for c in old['cases']}
    rows = []
    start = time.monotonic()
    cache_receipt = read(out/'cache_receipt.json')
    def tensor(images):
        return torch.from_numpy(np.stack(images).astype(np.float32)/np.float32(255)).permute(0, 3, 1, 2).contiguous().cuda()
    models = {}
    for route in p['selected_routes']:
        model = build_model(root, parent, provider, root/'readouts.npz', route)
        state = torch.load(out/('state_'+route+'50/full_state.pth'), map_location='cpu', weights_only=True)
        assert state['route'] == route and state['route_updates'] == 50
        model.load_state_dict(state['model'], strict=True)
        assert state_hash(model) == read(out/('state_'+route+'50/receipt.json'))['model_state']
        assert not model.training and all(not v.requires_grad for v in model.parameters())
        models[route] = model
    for index in p['schedule'][:2]:
        block = p['reference_blocks'][index]
        cs = [cases[c] for c in block['case_ids']]
        ref = refs[block['reference']]
        images = [np.asarray(Image.open(parent/c['input']).convert('RGB')).copy() for c in cs]
        mask = np.asarray(Image.open(parent/ref['observed']).convert('L')) > 0
        x = tensor(images)
        mask_t = torch.from_numpy(np.stack([mask]*5).astype(np.float32))[:, None].cuda()
        grid = torch.from_numpy(np.stack([grid112(ref['matrix112'])]*5)).cuda()
        with torch.no_grad():
            maps = provider.current.observe(x)
            features = provider.conditioned_features(maps)
            proof = cache_receipt['rows'][index]
            import hashlib
            array_hash = lambda v: hashlib.sha256(np.ascontiguousarray(v.detach().cpu().numpy()).tobytes()).hexdigest()
            assert proof['maps'] == [array_hash(v) for v in maps]
            assert proof['features'] == {k: array_hash(v) for k, v in features.items()}
            for route, model in models.items():
                pred = torch.where(mask_t.bool(), model.forward_cached(x, maps, features), x)
                a = pred.cpu().permute(0, 2, 3, 1).numpy()
                png = np.stack([deliver(v, image, mask) for v, image in zip(a, images)])
                with np.load(out/'baseline/packs'/f'r{index:02d}.npz', allow_pickle=False) as saved_base:
                    baseline = saved_base['raw'].copy()
                baseline_t = torch.from_numpy(baseline).permute(0, 3, 1, 2).contiguous().cuda()
                baseline_png = [deliver(v, image, mask) for v, image in zip(baseline, images)]
                targets = [np.asarray(Image.open(parent/ref['target']).convert('RGB')).copy() for _ in cs]
                target_t = tensor(targets)
                raw_vec = identity.embedding(torch.cat([pred, baseline_t, target_t]),
                    torch.cat([mask_t]*3), torch.cat([grid]*3)).cpu().numpy()
                png_vec = identity.embedding(torch.cat([tensor(list(png)), tensor(baseline_png), target_t]),
                    torch.cat([mask_t]*3), torch.cat([grid]*3)).cpu().numpy()
                with np.load(out/(route+'50')/'packs'/f'r{index:02d}.npz', allow_pickle=False) as saved:
                    assert np.array_equal(a, saved['raw']), 'Fresh manual CUDA RAW replay differs; retain stop'
                    assert np.array_equal(raw_vec, saved['embeddings_raw'])
                    assert np.array_equal(png_vec, saved['embeddings_png'])
                for image, c in zip(png, cs):
                    assert np.array_equal(image, np.asarray(Image.open(out/(route+'50')/'png'/(c['id']+'.png')).convert('RGB')))
                rows.append(dict(route=route, reference=block['reference'], cases=5,
                    RAW_exact=True, PNG_exact=True, embedding_RAW_exact=True, embedding_PNG_exact=True,
                    fresh_frozen_feature_cache_hashes_exact=True))
        assert time.monotonic()-start <= 300, 'Fresh replay finite time stop'
        assert torch.cuda.max_memory_allocated() <= p['budgets']['peak_VRAM_bytes']
        import resource
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= p['budgets']['peak_RSS_bytes']
        assert shutil.disk_usage(root).free >= p['budgets']['disk_reserve_bytes']
    assert state_hash(provider) == p['feature_provider_initial_state']
    write(out/'fresh_VM_replay.json', dict(complete=True, rows=rows, replayed_cases=20,
        seconds=time.monotonic()-start, cap_seconds=300, optimizer_updates=0, gradient_queries=0,
        local_neural_construction=False, full_saved_output_audit_still_required=True, model_qualified=False))
    print(json.dumps(dict(fresh_manual_VM_replay=True, cases=20, optimizer_updates=0)), flush=True)


def export(root, parent, p, pin):
    machine_guard(root)
    start = time.monotonic()
    out = root/STAGE
    assert out.is_dir(), 'No training evidence exists; export does not launch training'
    archive = Path.home()/(STEM+'-results.tar.gz')
    partial = Path.home()/(STEM+'-results.tar.gz.partial')
    receipt = Path.home()/(STEM+'-export.json')
    sidecar = Path.home()/(STEM+'-results.tar.gz.sha256')
    assert not any(path.exists() for path in [archive, partial, receipt, sidecar]), 'Immutable prior export; preserve it'
    selected = [root/name for name in p['assets_sha256']] + [root/'protocol.json']
    selected += [path for path in out.rglob('*') if path.is_file()]
    selected += [root/name for name in ['trainer.log', 'training_exit.json', 'replay.log', 'replay_exit.json', 'saved_audit.log', 'saved_audit_exit.json'] if (root/name).is_file()]
    assert len(selected) == len(set(selected))
    assert all(not path.is_symlink() and path.resolve().is_relative_to(root) for path in selected)
    bytes_total = sum(path.stat().st_size for path in selected)
    assert bytes_total <= p['budgets']['output_byte_cap']+16*1024**2, 'Export input cap stop'
    assert shutil.disk_usage(root).free >= bytes_total+p['budgets']['disk_reserve_bytes']+64*1024**2, 'Export reserve stop'
    manifest_path = root/'result_manifest.json'
    write(manifest_path, dict(protocol_sha256=pin, files={path.relative_to(root).as_posix():
        dict(bytes=path.stat().st_size, sha256=sha(path)) for path in sorted(selected)},
        scientific_pass_not_implied=True, full_corpus_epochs=0))
    selected.append(manifest_path)
    with partial.open('xb') as raw_stream:
        with tarfile.open(fileobj=raw_stream, mode='w:gz', compresslevel=1) as archive_stream:
            for path in sorted(selected):
                assert time.monotonic()-start < 600, 'Finite export time stop; retain partial file'
                assert shutil.disk_usage(root).free >= p['budgets']['disk_reserve_bytes'], 'Export disk reserve stop'
                archive_stream.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    assert time.monotonic()-start < 600, 'Finite export completion stop'
    partial.rename(archive)
    digest = sha(archive)
    with sidecar.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(digest+'  '+archive.name+'\n')
    value = dict(complete=True, archive_sha256=digest, bytes=archive.stat().st_size,
        protocol_sha256=pin, seconds=time.monotonic()-start, training_success_not_implied=True,
        model_qualified=False, run_results_present=(out/'run_results.json').exists(),
        failure_present=(out/'failure.json').exists() or any(out.glob('quality_stop_*.json')),
        optimizer_updates=sum(read(out/('fit_'+route+'.json'))['updates'] for route in ['A', 'B'] if (out/('fit_'+route+'.json')).exists()))
    write(receipt, value)
    print(json.dumps(value), flush=True)


def child(root, flag, pin, seconds, log_name):
    """Live binary tee; a partial stdout line cannot block timeout handling."""
    import selectors
    machine_guard(root)
    command = [sys.executable, '-B', '-u', str(root/'scripts/cctv_dgp_spatial_readout_v1_vm.py'),
               '--root', str(root), '--protocol-sha', pin, flag]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               bufsize=0, start_new_session=True)
    print(json.dumps(dict(stage=flag, child_pid=process.pid, log=log_name, external_cap_seconds=seconds)), flush=True)
    start = time.monotonic()
    timeout, term_sent, kill_sent, eof = False, None, False, False
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    with (root/log_name).open('xb') as log:
        while not (eof and process.poll() is not None):
            now = time.monotonic()
            if now-start > seconds and process.poll() is None and term_sent is None:
                timeout, term_sent = True, now
                os.killpg(process.pid, signal.SIGTERM)
            if term_sent is not None and now-term_sent > 30 and process.poll() is None and not kill_sent:
                os.killpg(process.pid, signal.SIGKILL)
                kill_sent = True
            for key, _ in selector.select(.2):
                data = os.read(key.fileobj.fileno(), 65536)
                if not data:
                    eof = True
                    selector.unregister(key.fileobj)
                else:
                    log.write(data)
                    log.flush()
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
    selector.close()
    code = process.wait(timeout=30)
    return dict(flag=flag, exit_code=code, timeout=timeout, seconds=time.monotonic()-start,
                child_pid=process.pid, maximum_seconds=seconds, kill_grace_seconds=30, kill_sent=kill_sent)


def supervise(root, pin):
    machine_guard(root, idle=True)
    assert not (root/'outputs').exists() and not (root/'trainer.log').exists(), 'Used pilot; preserve it'
    training = child(root, '--train', pin, 3030, 'trainer.log')
    write(root/'training_exit.json', training)
    checks = []
    if (root/STAGE/'run_results.json').exists():
        fresh = child(root, '--replay', pin, 330, 'replay.log')
        write(root/'replay_exit.json', fresh)
        checks.append(fresh)
        audit = child(root, '--audit-saved', pin, 600, 'saved_audit.log')
        write(root/'saved_audit_exit.json', audit)
        checks.append(audit)
    exported = child(root, '--export', pin, 630, 'export.log')
    write(root/'supervisor_receipt.json', dict(complete=True, training=training, checks=checks, export=exported,
        training_success_not_implied=True, model_qualified=False, automatic_follow_on=False))
    print(json.dumps(dict(training_exit=training['exit_code'], export_exit=exported['exit_code'],
                         check_exits=[v['exit_code'] for v in checks], model_qualified=False)), flush=True)
    return 0 if exported['exit_code'] == 0 and training['exit_code'] in [0, 2] and all(v['exit_code'] == 0 for v in checks) else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent-root', type=Path)
    parser.add_argument('--protocol-sha', required=True)
    flags = parser.add_mutually_exclusive_group(required=True)
    for name in ['verify-transfer', 'train', 'replay', 'audit-saved', 'export', 'supervise']:
        flags.add_argument('--'+name, action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    parent = args.parent_root.resolve() if args.parent_root else (root.parent/PARENT_NAME).resolve()
    if not args.verify_transfer:
        machine_guard(root)
    p, old = verify_transfer(root, parent, args.protocol_sha)
    if args.verify_transfer:
        print(json.dumps(dict(complete=True, assets=len(p['assets_sha256']), parent_assets=len(p['parent_required_sha256']),
            cases=100, neural_calls=0, optimizer_updates=0, gradient_queries=0)), flush=True)
        return 0
    if args.train:
        return train(root, parent, p, old, args.protocol_sha)
    if args.replay:
        replay(root, parent, p, old)
        return 0
    if args.audit_saved:
        from audit_cctv_dgp_spatial_readout_v1_return import audit_saved
        audit_saved(root, parent, p, old, root/STAGE/'saved_audit_vm.json')
        return 0
    if args.export:
        export(root, parent, p, args.protocol_sha)
        return 0
    return supervise(root, args.protocol_sha)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

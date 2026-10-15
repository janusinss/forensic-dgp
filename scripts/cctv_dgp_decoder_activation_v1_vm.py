"""Manual L4 measurement of retained states/direction; zero optimizer or proposals."""
import argparse
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
from cctv_dgp_decoder_activation_v1_contract import (
    NAME, STEM, STAGE, REPAIR_NAME, read, write, sha, array_sha, require_manual, verify_transfer,
)
from cctv_dgp_decoder_activation_v1_context import (
    MeasurementContext, source_cwd, file_module, SOURCE_NAME, PARENT_NAME, REPLAY_NAME, REPAIR_STAGE,
)

PROGRESS_CAP = 2*1024**2


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


def run(root, repair, source, parent, replay, p, repaired, old, data, pin):
    machine_guard(root, idle=True)
    assert not (root / 'outputs').exists(), 'Immutable completed/partial diagnostic; preserve it'
    out = root / STAGE
    out.mkdir(parents=True)
    start, ctx = time.monotonic(), None
    counts = dict(optimizer_updates=0, gradient_queries=0, backwards=0,
                  original_batch_calls=0, prior_feature_batch_calls=0,
                  decoder_batch_calls=0, recognizer_batch_calls=0)
    progress = dict(stage='resource preflight', state=None, reference=None)
    completed_states, completed_gradients = [], []

    def ensure_write_budget(additional):
        current = sum(v.stat().st_size for v in out.rglob('*') if v.is_file())
        assert current+additional <= p['budgets']['output_byte_cap']-p['budgets']['emergency_metadata_bytes'], 'Preserve emergency evidence budget'
        assert shutil.disk_usage(root).free >= additional+p['budgets']['disk_reserve_bytes'], 'Preserve disk reserve before write'

    def clock():
        assert time.monotonic()-start <= p['budgets']['worker_seconds'], 'Finite worker stop'
        assert shutil.disk_usage(root).free >= p['budgets']['disk_reserve_bytes'], 'Disk reserve stop'
        assert sum(v.stat().st_size for v in out.rglob('*') if v.is_file()) <= p['budgets']['output_byte_cap']-p['budgets']['emergency_metadata_bytes'], 'Output cap stop with emergency reserve'
        for key, value in counts.items():
            assert value <= p['budgets'].get('maximum_'+key, p['budgets'].get(key)), key + ' finite count stop'
        if ctx is not None and hasattr(ctx, 'torch'):
            import resource
            ctx.torch.cuda.synchronize()
            assert ctx.torch.cuda.max_memory_allocated() <= p['budgets']['peak_VRAM_bytes'], 'VRAM stop'
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= p['budgets']['peak_RSS_bytes'], 'RSS stop'
        journal = out/'progress.jsonl'
        line = json.dumps(dict(seconds=time.monotonic()-start, progress=progress, counts=dict(counts)), allow_nan=False)+'\n'
        assert (journal.stat().st_size if journal.exists() else 0)+len(line.encode('utf-8')) <= PROGRESS_CAP, 'Finite progress journal cap'
        with journal.open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(line)

    def interrupted(_signal, _frame):
        raise RuntimeError('Finite supervisor stop; preserve partial diagnostic')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        mem = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
        free = shutil.disk_usage(root).free
        assert PROGRESS_CAP < p['budgets']['emergency_metadata_bytes']
        # Bounds assume no compression benefit: 400 RAW arrays and delivered
        # PNGs, 70 full float32 gradients, two clipped controls, two full states,
        # retained vectors/float64 direction, embedding/control metadata margin.
        estimate = (400*256*256*3*5 + 70*609219*4 + 2*609219*4
                    + 2*(source/'outputs/spatial_readout_v1/state_A0/full_state.pth').stat().st_size
                    + 6*609219*8 + 32*1024**2)
        write(out / 'storage_start.json', dict(free_bytes=free, MemAvailable_bytes=mem['MemAvailable'], projected_output_bytes=estimate,
            output_cap_bytes=p['budgets']['output_byte_cap'], emergency_metadata_bytes=p['budgets']['emergency_metadata_bytes'],
            neural_calls=0, gradient_queries=0, optimizer_updates=0))
        assert free >= p['budgets']['minimum_free_bytes'] and mem['MemAvailable'] >= p['budgets']['available_RAM_before_import_bytes']
        assert estimate <= p['budgets']['output_byte_cap']
        assert free >= 2*estimate + p['budgets']['disk_reserve_bytes'], 'Preserve export reserve before models'
        ctx = MeasurementContext.__new__(MeasurementContext)
        ctx.__init__(root, repair, source, parent, replay, p, repaired, old, data, out, counts, clock)
        np, torch = ctx.np, ctx.torch
        math_module = file_module(root/'cctv_dgp_decoder_activation_v1_math.py', 'frozen_activation_math',
                                  p['assets_sha256']['cctv_dgp_decoder_activation_v1_math.py'])
        write(out/'environment.json', dict(host=os.uname().nodename, GPU=torch.cuda.get_device_name(0), torch=torch.__version__,
            CUDA=torch.version.cuda, python=sys.version,
            packages={name: importlib.metadata.version(name) for name in ['torch','torchvision','numpy','Pillow','scipy','scikit-image','onnx','onnx2torch']},
            DGP_provenance=ctx.provenance, frozen_states=ctx.frozen_start, historical_training_called=False,
            corrected_helper_path=str(Path(ctx.corrected.__file__).resolve()), corrected_helper_sha256=sha(ctx.corrected.__file__),
            peak_includes_dependency_loading=True,
            optimizer_present=False, new_parameter_proposals=0, new_amplitudes=0))
        np.save(out/'retained_direction.npy', ctx.direction, allow_pickle=False)
        write(out/'retained_direction.json', dict(kind='retained_float32_A1_minus_initial_promoted_to_float64',
            direction_sha256=array_sha(ctx.direction), dtype=str(ctx.direction.dtype), elements=ctx.direction.size,
            norm_L2=float(np.linalg.norm(ctx.direction)), new_parameter_proposal=False))
        with source_cwd(source):
            ensure_write_budget((source/'outputs/spatial_readout_v1/state_A0/full_state.pth').stat().st_size+2*1024**2)
            ctx.save_state('before', pin)
            for state in p['state_labels']:
                progress.update(stage='fresh batch-five retained-state replay', state=state)
                began = time.monotonic()
                folder = out/state
                folder.mkdir()
                (folder/'packs').mkdir()
                (folder/'png').mkdir()
                ctx.assign(state)
                records = []
                with torch.no_grad():
                    for index, b in enumerate(ctx.caches):
                        block = p['reference_blocks'][index]
                        progress['reference'] = block['reference']
                        g = ctx.gpu(b)
                        pred = ctx.forward(g)
                        value, current, baseline, mean_shift, embedding, record = ctx.objective(pred, g, b)
                        raw = pred.cpu().permute(0, 2, 3, 1).numpy().copy()
                        delivered = np.stack([ctx.deliver(a, c, m) for a, c, m in zip(raw, b['camera'], b['mask8'])])
                        control = repair/REPAIR_STAGE/state
                        with np.load(control/'packs'/f'r{index:02d}.npz', allow_pickle=False) as saved:
                            assert saved['ids'].tolist() == block['case_ids']
                            assert np.array_equal(raw, saved['raw']), 'Fresh retained-state RAW differs'
                        assert all(np.array_equal(v, ctx.pixels(control/'png'/(c['id']+'.png'))) for v, c in zip(delivered, b['cases'])), 'Fresh retained-state floor-PNG differs'
                        ensure_write_budget(5*256*256*3*5+1024**2)
                        for i, case in enumerate(b['cases']):
                            assert np.array_equal(raw[i][~b['mask8'][i]], b['camera'][i][~b['mask8'][i]].astype(np.float32)/np.float32(255))
                            assert np.array_equal(delivered[i][~b['mask8'][i]], b['camera'][i][~b['mask8'][i]])
                            ctx.Image.fromarray(delivered[i]).save(folder/'png'/(case['id']+'.png'))
                        np.savez_compressed(folder/'packs'/f'r{index:02d}.npz', ids=np.asarray(block['case_ids']), raw=raw,
                                            embedding_batch5=embedding.cpu().numpy(), target_embeddings=b['target_vectors'])
                        finite = math_module.finite_objective(record['current'], record['baseline'], old['normalizers'],
                            old['active_anchor_normalizers'], photometric=record['per_image_photometric'], reconstruction_weights=old['reconstruction_weights'])
                        delta = float(finite['objective']) - record['objective']
                        assert abs(delta) <= p['objective_replay_max_abs'], 'Saved arithmetic differs from batch-five objective'
                        records.append(dict(reference=block['reference'], source=block['source'], case_ids=block['case_ids'],
                            raw_control_exact=True, floor_PNG_control_exact=True, **record, finite_arithmetic=finite,
                            finite_minus_torch_objective=delta, raw_sha256=array_sha(raw), delivered_uint8_sha256=array_sha(delivered)))
                        del g, pred, value, current, baseline, mean_shift, embedding, raw, delivered
                        clock()
                        assert time.monotonic()-began <= p['budgets']['snapshot_seconds'], 'Finite retained-state replay stop'
                retained = control/'objective_components.json'
                shutil.copyfile(retained, folder/'retained_batch15_objective_components.json')
                write(folder/'objective_components.json', dict(complete=True, rows=records, recognizer_batch=5,
                    historical_batch15_separate=True, historical_components_sha256=sha(retained),
                    state_vector_sha256=array_sha(ctx.vectors[state]), model_state=ctx.state_hash(ctx.model),
                    model_qualified=False, seconds=time.monotonic()-began))
                completed_states.append(state)
                if len(completed_states) == 1:
                    elapsed = time.monotonic()-began
                    remaining_states = len(p['state_labels'])-1
                    projected = time.monotonic()-start + remaining_states*elapsed*1.5 + p['budgets']['gradient_seconds'] + 30
                    write(out/'state_timing_projection.json', dict(completed_states=1, seconds=elapsed,
                        remaining_states=remaining_states, snapshot_multiplier=1.5,
                        remaining_gradient_bound_seconds=p['budgets']['gradient_seconds'],
                        projected_worker_seconds=projected, worker_cap_seconds=p['budgets']['worker_seconds']))
                    assert projected <= p['budgets']['worker_seconds'], 'First retained-state projection exceeds finite worker cap'
                print(json.dumps(dict(retained_state=state, exact_RAW_PNG_cases=100, recognizer_batch=5)), flush=True)
            ctx.assign('initial')
            ctx.model.enable_learning()
            selected = ctx.model.learning_parameters()
            gradient_root = out/'gradients'
            gradient_root.mkdir()
            began = time.monotonic()
            for ordinal, index in enumerate(p['derivative_reference_indices']):
                b, block = ctx.caches[index], p['reference_blocks'][index]
                progress.update(stage='initial underlying scalar VJPs', state='initial', reference=block['reference'])
                folder = gradient_root/f'r{index:02d}'
                folder.mkdir()
                g = ctx.gpu(b)
                with torch.enable_grad():
                    pred = ctx.forward(g)
                    loss, current, baseline, mean_shift, embedding, record = ctx.objective(pred, g, b)
                    assert np.array_equal(pred.detach().cpu().permute(0, 2, 3, 1).numpy(), b['baseline'])
                    scalars = ([current['MSE'][i] for i in range(5)] + [current['SSIM_loss'][i] for i in range(5)]
                        + [current['ArcFace_loss'][i] for i in range(5)] + [current['landmark_structure'][i] for i in range(1,5)]
                        + [mean_shift[i, channel] for i in range(5) for channel in range(3)] + [loss])
                    assert len(scalars) == 35
                    rows, scalar_values = [], []
                    for position, scalar in enumerate(scalars):
                        derivatives = torch.autograd.grad(scalar, selected, retain_graph=position < 34, allow_unused=False)
                        counts['gradient_queries'] += 1
                        assert all(bool(torch.isfinite(v).all()) for v in derivatives)
                        row = np.concatenate([v.detach().cpu().numpy().reshape(-1) for v in derivatives]).astype(np.float32)
                        assert row.shape == (609219,) and np.isfinite(row).all()
                        rows.append(row)
                        scalar_values.append(float(scalar.detach()))
                        if position == 34:
                            # Reproduce the numerical clipping operation without
                            # installing .grad, backward(), or an optimizer.
                            norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(v, 2) for v in derivatives]), 2)
                            factor = torch.clamp(1. / (norm + 1e-6), max=1.)
                            clipped = np.concatenate([(v.detach()*factor).cpu().numpy().reshape(-1) for v in derivatives]).astype(np.float32)
                            full_norm, clip_factor = float(norm.detach()), float(factor.detach())
                        del derivatives, row
                        clock()
                        assert time.monotonic()-began <= p['budgets']['gradient_seconds'], 'Finite gradient measurement stop'
                matrix = np.stack(rows)
                del rows
                slopes = matrix.astype(np.float64) @ ctx.direction
                assert np.isfinite(slopes).all() and slopes.shape == (35,)
                ensure_write_budget(matrix.nbytes+clipped.nbytes+2*1024**2)
                np.save(folder/'gradients.npy', matrix, allow_pickle=False)
                np.save(folder/'clipped_full_gradient.npy', clipped, allow_pickle=False)
                gradient_embedding = embedding.detach().cpu().numpy().copy()
                assert array_sha(gradient_embedding) == record['embedding_sha256']
                np.save(folder/'embedding_batch5.npy', gradient_embedding, allow_pickle=False)
                current_slopes = dict(MSE=slopes[:5].tolist(), SSIM_loss=slopes[5:10].tolist(),
                    ArcFace_loss=slopes[10:15].tolist(), landmark_structure=[0.] + slopes[15:19].tolist())
                channel_slopes = slopes[19:34].reshape(5,3)
                right = math_module.mean_control_right_mse_at_initial(
                    g['baseline'].detach().cpu().numpy(), g['target'].detach().cpu().numpy(),
                    g['mask'].detach().cpu().numpy(), channel_slopes)
                photo_slopes = dict(actual_MSE=current_slopes['MSE'], mean_only_MSE=right['mean_only_MSE_slopes'])
                one_sided = math_module.one_sided_objective(record['current'], record['baseline'], current_slopes,
                    old['normalizers'], old['active_anchor_normalizers'], photometric=record['per_image_photometric'],
                    photometric_slopes=photo_slopes, mean_control_derivative_kind='idealized_pixel_clip',
                    reconstruction_weights=old['reconstruction_weights'])
                replay_control = None
                if ordinal == 0:
                    with np.load(repair/REPAIR_STAGE/'trajectory/update_001.npz', allow_pickle=False) as saved:
                        retained = saved['clipped_gradient'].copy()
                    a, z = clipped.astype(np.float64), retained.astype(np.float64)
                    maximum, l2 = float(np.max(np.abs(a-z))), float(np.linalg.norm(a-z))
                    cosine = float(a@z/(np.linalg.norm(a)*np.linalg.norm(z)))
                    bound = p['clipped_gradient_replay_bound']
                    replay_control = dict(max_abs=maximum, difference_L2=l2, cosine=cosine,
                        fresh_clipped_L2=float(np.linalg.norm(a)), retained_clipped_L2=float(np.linalg.norm(z)),
                        bound=bound, pass_=maximum <= bound['max_abs'] and cosine >= bound['cosine_min'],
                        full_gradient_norm_before_clip=full_norm, clip_factor=clip_factor,
                        actual_fitting_reference=True, quality_gate_relaxed=False)
                    write(folder/'clipped_gradient_replay.json', replay_control)
                    assert replay_control['pass_'], 'Underlying batch-five clipped gradient replay differs; retain diagnostic stop'
                write(folder/'measurement.json', dict(complete=True, reference=block['reference'], source=block['source'],
                    case_ids=block['case_ids'], scalar_labels=p['gradient_layout'], scalar_values=scalar_values,
                    directional_slopes=slopes.tolist(), gradients_shape=list(matrix.shape), gradients_dtype=str(matrix.dtype),
                    gradients_sha256=array_sha(matrix), clipped_gradient_sha256=array_sha(clipped),
                    embedding_batch5_file='embedding_batch5.npy',
                    embedding_batch5_file_sha256=sha(folder/'embedding_batch5.npy'),
                    embedding_batch5_array_sha256=array_sha(gradient_embedding),
                    retained_direction_sha256=array_sha(ctx.direction), selected_decoder_branch_only=True,
                    branch_finite_validation='only the three retained nonzero amplitudes; no universal derivative claim',
                    current_slopes=current_slopes, unqueried_clear_structure_slope_placeholder=0.,
                    channel_mean_slopes=channel_slopes.tolist(), photometric_slopes=photo_slopes,
                    mean_control_right_derivative=right, one_sided=one_sided, **record,
                    full_gradient_norm_before_clip=full_norm, clip_factor=clip_factor, clipped_gradient_replay=replay_control,
                    literal_float32_step_derivative_claim=False, optimizer_updates=0, backwards=0))
                completed_gradients.append(block['reference'])
                del g, pred, loss, current, baseline, mean_shift, embedding, gradient_embedding, scalars, matrix, slopes, clipped
                clock()
                assert time.monotonic()-began <= p['budgets']['gradient_seconds']
                if ordinal == 0:
                    elapsed = time.monotonic()-began
                    projected_gradient = 2*elapsed*1.75
                    projected_worker = time.monotonic()-start + elapsed*1.75 + 30
                    write(out/'gradient_timing_projection.json', dict(completed_references=1, gradient_queries=35,
                        seconds=elapsed, gradient_multiplier=1.75, projected_gradient_seconds=projected_gradient,
                        gradient_cap_seconds=p['budgets']['gradient_seconds'], projected_worker_seconds=projected_worker,
                        worker_cap_seconds=p['budgets']['worker_seconds']))
                    assert projected_gradient <= p['budgets']['gradient_seconds'], 'First scalar-gradient projection exceeds finite gradient cap'
                    assert projected_worker <= p['budgets']['worker_seconds'], 'First scalar-gradient projection exceeds finite worker cap'
                print(json.dumps(dict(gradient_reference=block['reference'], queries=counts['gradient_queries'], optimizer_updates=0)), flush=True)
            # Detect forbidden buffer/frozen changes before restoration could
            # conceal them. The exception path saves the actual stopped state.
            assert ctx.buffer_hash(ctx.model) == ctx.initial_buffer_hash
            assert ctx.state_hash({k: v for k, v in ctx.model.state_dict().items() if k not in ctx.names}) == ctx.fixed_hash
            assert ctx.frozen_start == dict(original=ctx.state_hash(ctx.original.net),
                recognizer=ctx.state_hash(ctx.identity), provider=ctx.state_hash(ctx.provider))
            restored = ctx.restore()
            write(out/'restoration.json', restored)
            assert restored['RNG_unchanged_before_restore'], 'Unexpected RNG consumption; retained initial RNG restored'
            ensure_write_budget((source/'outputs/spatial_readout_v1/state_A0/full_state.pth').stat().st_size+2*1024**2)
            ctx.save_state('after', pin)
        assert counts == dict(optimizer_updates=0, gradient_queries=70, backwards=0, original_batch_calls=20,
            prior_feature_batch_calls=20, decoder_batch_calls=82, recognizer_batch_calls=82)
        clock()
        write(out/'run_results.json', dict(complete=True, counts=counts, state_labels=completed_states,
            derivative_reference_ids=completed_gradients, exact_retained_RAW_PNG_cases=400,
            batch_five_objective_reference_states=80, full_parameter_gradients=70,
            parameters_per_gradient=609219, recognizer_batch=5, historical_batch15_separate=True,
            initial_full_state_and_RNG_restored=True, frozen_models_unchanged=True,
            new_parameter_proposals=0, new_amplitudes=0, full_corpus_epochs=0,
            model_qualified=False, new_incumbent_selection=False, automatic_follow_on=False,
            failed_recipe_resume_allowed=False, seconds=time.monotonic()-start))
        require_manual(root)
        return 0
    except BaseException as exc:
        save_error = restore_error = None
        selected_receipt = None
        if ctx is not None and hasattr(ctx, 'model'):
            try:
                selected = ctx.flat()
                selected_path = out/'technical_stop_selected_vector.npy'
                assert not selected_path.exists()
                current_size = sum(v.stat().st_size for v in out.rglob('*') if v.is_file())
                assert current_size+selected.nbytes+1024**2 <= p['budgets']['output_byte_cap']
                assert shutil.disk_usage(root).free >= selected.nbytes+1024**2
                ctx.np.save(selected_path, selected, allow_pickle=False)
                selected_receipt = dict(path=selected_path.relative_to(root).as_posix(), sha256=sha(selected_path),
                    array_sha256=array_sha(selected), model_state=ctx.state_hash(ctx.model), buffer_state=ctx.buffer_hash(ctx.model),
                    exact_current_selected_vector_saved=True, model_qualified=False)
                write(out/'technical_stop_selected_state.json', selected_receipt)
                del selected
                # The initial full state and before-state are already retained.
                # A third model snapshot is optional, never allowed to consume
                # the output cap or the reserved free disk needed by export.
                expected = (source/'outputs/spatial_readout_v1/state_A0/full_state.pth').stat().st_size+2*1024**2
                current_size = sum(v.stat().st_size for v in out.rglob('*') if v.is_file())
                if current_size+expected+1024**2 <= p['budgets']['output_byte_cap'] and shutil.disk_usage(root).free >= expected+p['budgets']['disk_reserve_bytes']:
                    ctx.save_state('technical_stop', pin)
                else:
                    save_error = 'Optional full emergency snapshot omitted to preserve output/disk export reserve; exact current selected vector and state hashes retained'
            except BaseException as stopped:
                save_error = repr(stopped)
            if hasattr(ctx, 'CPU_rng'):
                try:
                    with source_cwd(source):
                        restored = ctx.restore()
                    write(out/'failure_restoration.json', restored)
                except BaseException as stopped:
                    restore_error = repr(stopped)
        write(out/'failure.json', dict(complete=False, error_type=type(exc).__name__, message=str(exc),
            traceback=traceback.format_exc(), progress=progress, counts=counts, completed_states=completed_states,
            completed_gradient_references=completed_gradients, failure_state_save_error=save_error,
            current_selected_state_receipt=selected_receipt, restoration_error=restore_error,
            model_qualified=False, failed_recipe_resume_allowed=False,
            seconds=time.monotonic()-start))
        raise


def export(root, p, pin):
    require_manual(root)
    began = time.monotonic()
    manifest = root/'result_manifest.json'
    archive = Path.home()/(STEM+'-results.tar.gz')
    sidecar, receipt, partial = Path(str(archive)+'.sha256'), Path.home()/(STEM+'-export.json'), Path(str(archive)+'.partial')
    assert all(not value.exists() for value in (manifest, archive, sidecar, receipt, partial)), 'Preserve existing complete/partial export'
    files = [root/name for name in p['assets_sha256']] + [root/'protocol.json']
    files += sorted(value for value in (root/STAGE).rglob('*') if value.is_file())
    files += [root/name for name in ('run.log','run_exit.json','saved_audit.log','saved_audit_exit.json') if (root/name).is_file()]
    assert len(files) == len(set(files)) and all(not v.is_symlink() and v.resolve().is_relative_to(root) for v in files)
    total = sum(value.stat().st_size for value in files)
    assert total <= p['budgets']['output_byte_cap']+16*1024**2
    assert shutil.disk_usage(root).free >= total+p['budgets']['disk_reserve_bytes']
    result_path, failure_path = root/STAGE/'run_results.json', root/STAGE/'failure.json'
    result = None
    receipt_read_errors = []
    for candidate in (result_path, failure_path):
        if not candidate.is_file():
            continue
        try:
            value = read(candidate)
            queries = value['counts']['gradient_queries']
            assert isinstance(queries, int) and not isinstance(queries, bool) and 0 <= queries <= p['budgets']['maximum_gradient_queries']
            result = value
            break
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, AssertionError) as invalid:
            # SIGKILL can interrupt a JSON write as well as prevent it. Retain
            # the opaque partial file, report unknown exact counts, and export.
            receipt_read_errors.append(dict(path=candidate.relative_to(root).as_posix(), error=repr(invalid)))
    last_progress = None
    progress_path = root/STAGE/'progress.jsonl'
    if progress_path.is_file():
        for line in progress_path.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A killed final write can be incomplete opaque evidence.
            if isinstance(row, dict) and isinstance(row.get('counts'), dict):
                last_progress = row
    terminal = read(root/'run_exit.json') if (root/'run_exit.json').is_file() else None
    assert result is not None or terminal is not None, 'Need a worker receipt or authoritative terminal supervisor receipt'
    exact_gradient_queries = result['counts']['gradient_queries'] if result is not None else None
    partial_without_receipt = result is None
    bindings = {}
    for path in files:
        bindings[path.relative_to(root).as_posix()] = dict(bytes=path.stat().st_size, sha256=sha(path))
        assert time.monotonic()-began <= p['budgets']['export_seconds']
    write(manifest, dict(complete=True, protocol_sha256=pin, files=bindings, optimizer_updates=0,
        gradient_queries=exact_gradient_queries, gradient_query_count_exact=result is not None,
        last_known_progress=last_progress, terminal_worker=terminal,
        partial_terminal_without_worker_receipt=partial_without_receipt,
        incomplete_worker_receipt_errors=receipt_read_errors,
        model_qualified=False, training_success_not_implied=True))
    with tarfile.open(partial, 'x:gz', compresslevel=1) as stream:
        for path in [manifest]+files:
            assert time.monotonic()-began <= p['budgets']['export_seconds']
            assert shutil.disk_usage(root).free >= p['budgets']['disk_reserve_bytes']
            stream.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    digest = sha(partial)
    assert time.monotonic()-began <= p['budgets']['export_seconds']
    assert shutil.disk_usage(root).free >= p['budgets']['disk_reserve_bytes'] and not archive.exists()
    partial.rename(archive)
    with sidecar.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(digest+'  '+archive.name+'\n')
    write(receipt, dict(complete=True, protocol_sha256=pin, archive_sha256=digest, bytes=archive.stat().st_size,
        seconds=time.monotonic()-began, optimizer_updates=0, gradient_queries=exact_gradient_queries,
        gradient_query_count_exact=result is not None, last_known_progress=last_progress,
        terminal_worker=terminal, partial_terminal_without_worker_receipt=partial_without_receipt,
        incomplete_worker_receipt_errors=receipt_read_errors,
        model_qualified=False, training_success_not_implied=True,
        run_results_present=result_path.is_file(), failure_present=failure_path.is_file()))
    print(json.dumps(read(receipt)), flush=True)


def child(root, repair, source, parent, replay, pin, flag, seconds, log_name):
    import selectors
    args = [sys.executable, '-B', '-u', str(root/'scripts/cctv_dgp_decoder_activation_v1_vm.py'),
        '--root', str(root), '--repair-root', str(repair), '--source-root', str(source),
        '--parent-root', str(parent), '--replay-root', str(replay), '--protocol-sha', pin, flag]
    began, timeout, killed = time.monotonic(), False, False
    with (root/log_name).open('xb') as log, selectors.DefaultSelector() as selector:
        process = subprocess.Popen(args, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, bufsize=0)
        print(json.dumps(dict(flag=flag, child_pid=process.pid, log=log_name, external_cap_seconds=seconds)), flush=True)
        selector.register(process.stdout, selectors.EVENT_READ)
        while selector.get_map() or process.poll() is None:
            elapsed = time.monotonic()-began
            if elapsed > seconds and not timeout and process.poll() is None:
                timeout = True
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            if elapsed > seconds+30 and process.poll() is None and not killed:
                killed = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            for key, _ in selector.select(.2):
                value = os.read(key.fileobj.fileno(), 65536)
                if value:
                    log.write(value)
                    log.flush()
                    sys.stdout.buffer.write(value)
                    sys.stdout.buffer.flush()
                else:
                    selector.unregister(key.fileobj)
        code = process.wait()
    return dict(flag=flag, exit_code=code, timeout=timeout, kill_sent=killed, seconds=time.monotonic()-began,
                child_pid=process.pid, maximum_seconds=seconds, kill_grace_seconds=30)


def supervise(root, repair, source, parent, replay, p, repaired, old, data, pin):
    machine_guard(root, idle=True)
    reserved = ['outputs','result_manifest.json','run.log','run_exit.json','saved_audit.log','saved_audit_exit.json','export.log','supervisor_receipt.json']
    reserved += [str(Path.home()/(STEM+'-results.tar.gz'+suffix)) for suffix in ('','.partial','.sha256')]
    reserved += [str(Path.home()/(STEM+'-export.json'))]
    assert all(not (root/name).exists() for name in reserved), 'Used run/export; preserve evidence'
    worked = child(root, repair, source, parent, replay, pin, '--run', p['budgets']['external_seconds'], 'run.log')
    write(root/'run_exit.json', worked)
    checked = None
    if (root/STAGE/'run_results.json').is_file() or (root/STAGE/'failure.json').is_file():
        checked = child(root, repair, source, parent, replay, pin, '--audit-saved', p['budgets']['saved_audit_external_seconds'], 'saved_audit.log')
        write(root/'saved_audit_exit.json', checked)
    exported = child(root, repair, source, parent, replay, pin, '--export', p['budgets']['export_external_seconds'], 'export.log')
    write(root/'supervisor_receipt.json', dict(complete=True, run=worked, saved_audit=checked, export=exported,
        model_qualified=False, optimizer_updates=0, automatic_follow_on=False))
    return 0 if worked['exit_code'] == 0 and checked and checked['exit_code'] == 0 and exported['exit_code'] == 0 else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    for name in ('repair-root','source-root','parent-root','replay-root'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--protocol-sha', required=True)
    flags = parser.add_mutually_exclusive_group(required=True)
    for name in ('verify-transfer','run','audit-saved','export','supervise'):
        flags.add_argument('--'+name, action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    repair = args.repair_root.resolve() if args.repair_root else root.parent/REPAIR_NAME
    source = args.source_root.resolve() if args.source_root else root.parent/SOURCE_NAME
    parent = args.parent_root.resolve() if args.parent_root else root.parent/PARENT_NAME
    replay = args.replay_root.resolve() if args.replay_root else root.parent/REPLAY_NAME
    if not args.verify_transfer:
        require_manual(root)
    p, repaired, old, data = verify_transfer(root, repair, source, parent, replay, args.protocol_sha)
    if args.verify_transfer:
        print(json.dumps(dict(complete=True, assets=len(p['assets_sha256']), repair_assets=len(p['repair_required_sha256']),
            source_assets=len(p['source_required_sha256']), parent_assets=len(p['parent_required_sha256']),
            replay_assets=len(p['replay_required_sha256']), neural_calls=0, gradient_queries=0, optimizer_updates=0)))
    elif args.run:
        raise SystemExit(run(root, repair, source, parent, replay, p, repaired, old, data, args.protocol_sha))
    elif args.export:
        export(root, p, args.protocol_sha)
    elif args.audit_saved:
        from audit_cctv_dgp_decoder_activation_v1_return import audit_saved
        audit_saved(root, repair, source, parent, replay, p, repaired, old, data, root/STAGE/'independent_saved_audit_vm.json')
    else:
        raise SystemExit(supervise(root, repair, source, parent, replay, p, repaired, old, data, args.protocol_sha))


if __name__ == '__main__':
    main()

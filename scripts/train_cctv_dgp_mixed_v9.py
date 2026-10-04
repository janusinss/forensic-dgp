"""Finite mixed-source pixel-foundation pilot; all backward/updates on L4 VM."""
import argparse
import collections
import importlib.metadata
import math
from pathlib import Path
import platform
import shutil
import sys
import time

CONTEXT = {'out': None, 'model': None, 'updates': 0, 'backwards': 0, 'autograd': 0}


def train(root, expected_protocol_sha):
    sys.path.insert(0, str(root))
    from cctv_dgp_targets_v6 import require_vm
    require_vm(root)  # Before model construction, backward, optimizer or output creation.
    from cctv_dgp_mixed_v9 import START_STATE, fixed_loss_weights, group, read, sha, verify, write
    p = verify(root, expected_protocol_sha)
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from torch.utils.data import DataLoader
    from models import DGPSynthesizer
    from cctv_dgp_frozen_norm import install_frozen_instance_norm
    from cctv_dgp_pilot import (FixedObservedIdentity, PilotDataset, aggregate, buffer_hash,
        composite, exported_pixel_metrics, freeze_normalization, qualifies, state_hash)
    started = time.monotonic(); design = p['design']; deadline = started + 2400
    out = root / 'outputs/cctv_dgp_mixed_v9'
    assert not out.exists(), 'Preserve prior run; no automatic resume/repeat'
    out.mkdir(parents=True); CONTEXT['out'] = out
    torch.set_num_threads(4); torch.manual_seed(design['seed'])
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = False, False
    model = DGPSynthesizer().cuda().eval(); CONTEXT['model'] = model
    assert install_frozen_instance_norm(model) == 5
    starting = torch.load(root / p['weights']['start'], map_location='cpu', weights_only=True)
    model.load_state_dict(starting, strict=True)
    assert state_hash(model) == START_STATE
    buffers = buffer_hash(model)
    identity = FixedObservedIdentity(root / p['weights']['arcface'], 'cuda')
    teacher_state = state_hash(identity)
    refs = {r['id']: r for r in p['references']}
    fitting = PilotDataset(root, p, p['training_cases'])
    validation = PilotDataset(root, p, p['validation_cases'])
    train_indices = {c['id']: i for i, c in enumerate(p['training_cases'])}
    train_by_id = {c['id']: c for c in p['training_cases']}
    schedule = read(root / 'training_schedule_v9.json')['epochs']

    def clock():
        if time.monotonic() > deadline:
            raise TimeoutError('V9 trainer exceeded2400 seconds')

    def device(batch):
        return {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def errors(prediction, batch):
        return ((prediction - batch['target']).square() * batch['mask']).sum((1, 2, 3)) / (3 * batch['mask'].sum((1, 2, 3)))

    def save_state(path):
        assert not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, path)

    # Training-only calibration: frozen per-case raw MSE, with twenty pinned raw checks.
    calibration_started = time.monotonic(); calibration_rows = []
    calibration_raw_ids = set()
    for key in sorted({group(c) for c in p['training_cases']}):
        calibration_raw_ids.update(sorted(c['id'] for c in p['training_cases'] if group(c) == key)[:2])
    with torch.no_grad():
        for batch in DataLoader(fitting, batch_size=10, num_workers=0):
            clock(); batch = device(batch)
            prediction = composite(model(batch['low']), batch['low'], batch['mask'])
            values = errors(prediction, batch).cpu().tolist()
            for j, index in enumerate(batch['index'].tolist()):
                case = fitting.cases[index]
                row = {'case_id': case['id'], 'reference_id': case['reference_id'], 'group': group(case), 'MSE_raw': values[j]}
                if case['id'] in calibration_raw_ids:
                    name = 'calibration/raw/' + case['id'] + '.npy'
                    (out / name).parent.mkdir(parents=True, exist_ok=True)
                    np.save(out / name, prediction[j].permute(1, 2, 0).cpu().numpy(), allow_pickle=False)
                    row['raw_float'] = name
                assert math.isfinite(values[j]) and values[j] >= 0
                calibration_rows.append(row)
            if len(calibration_rows) % 500 < 10:
                print(f'training-only calibration {len(calibration_rows)}/3905 elapsed={time.monotonic()-started:.0f}s', flush=True)
    means = {key: math.fsum(r['MSE_raw'] for r in calibration_rows if r['group'] == key) / sum(r['group'] == key for r in calibration_rows)
             for key in sorted({r['group'] for r in calibration_rows})}
    weights = fixed_loss_weights(means)
    assert state_hash(model) == START_STATE and buffer_hash(model) == buffers
    write(out / 'calibration/weights.json', {'complete': True, 'training_only': True, 'rows': calibration_rows,
        'group_means': means, 'weights': weights, 'starting_state_hash': START_STATE,
        'raw_check_case_ids': sorted(calibration_raw_ids), 'validation_statistics_used': False,
        'seconds': time.monotonic() - calibration_started, 'formula': design['weight_rule']})
    calibration_sha = sha(out / 'calibration/weights.json')

    def objective(prediction, batch):
        values = errors(prediction, batch)
        keys = [source + '/' + profile for source, profile in zip(batch['source'], batch['profile'])]
        multipliers = torch.tensor([weights[k] for k in keys], device='cuda', dtype=values.dtype)
        return (values * multipliers).mean(), values

    first_indices = [[train_indices[c] for c in schedule['1'][0]]]
    batch = device(next(iter(DataLoader(fitting, batch_sampler=first_indices, num_workers=0))))
    loss, _ = objective(composite(model(batch['low']), batch['low'], batch['mask']), batch)
    gradients = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True); CONTEXT['autograd'] += 1
    active = [g for g in gradients if g is not None]
    assert active and all(torch.isfinite(g).all() for g in active) and sum(float(g.abs().sum()) for g in active) > 0
    assert all(param.grad is None for param in model.parameters())
    with torch.no_grad():
        model(batch['low'][:5])  # The calibration tail has five inputs; guard normalization.
    assert state_hash(model) == START_STATE and buffer_hash(model) == buffers and state_hash(identity) == teacher_state
    write(out / 'preflight.json', {'passed': True, 'autograd_calls': 1, 'optimizer_updates': 0,
        'active_gradient_tensors': len(active), 'batch_size': 10, 'tail_batch_size': 5,
        'cases': schedule['1'][0], 'starting_state_hash': START_STATE, 'buffers_unchanged': True,
        'calibration_sha256': calibration_sha, 'gpu': torch.cuda.get_device_name(0),
        'peak_allocated_cuda_bytes': torch.cuda.max_memory_allocated()})
    del gradients, active, loss, batch
    write(out / 'execution.json', {'protocol_sha256': expected_protocol_sha, 'source_sha256': sha(Path(__file__)),
        'host': platform.node(), 'gpu': torch.cuda.get_device_name(0), 'started_unix_seconds': time.time(),
        'torch': torch.__version__, 'starting_state_hash': START_STATE, 'buffer_hash': buffers,
        'teacher_state_hash': teacher_state, 'calibration_sha256': calibration_sha,
        'package_versions': {k: importlib.metadata.version(k) for k in ('torch', 'torchvision', 'numpy', 'Pillow', 'scikit-image', 'onnx2torch')},
        'runtime_cap_seconds': 2400, 'normalization_frozen': True, 'amp': False, 'ema': False,
        'validation_statistics_used_for_loss': False, 'native_used': False, 'production_promoted': False})
    shutil.copyfile(Path(__file__), out / 'executed_source.py')

    target_embeds = {}
    with torch.no_grad():
        for batch in DataLoader(validation, batch_size=10, num_workers=0):
            clock(); batch = device(batch)
            embedding = identity.embedding(batch['target'], batch['mask'], batch['grid']).cpu().numpy()
            for rid, value in zip(batch['reference_id'], embedding):
                if rid not in target_embeds:
                    target_embeds[rid] = value
                    file = out / 'target_embeddings' / (rid + '.npy'); file.parent.mkdir(exist_ok=True)
                    np.save(file, value, allow_pickle=False)
    assert len(target_embeds) == 104

    def evaluate(stage, include_input=False):
        stage_start = time.monotonic(); model.eval(); rows = []; input_rows = []
        for batch in DataLoader(validation, batch_size=10, num_workers=0):
            clock(); batch = device(batch)
            with torch.no_grad():
                prediction = composite(model(batch['low']), batch['low'], batch['mask'])
                raw = prediction.cpu().numpy().transpose(0, 2, 3, 1)
                rgb = np.clip(raw * 255, 0, 255).astype(np.uint8)
                support = batch['mask'].cpu().numpy()[:, 0] > 0
                for j, index in enumerate(batch['index'].tolist()):
                    input_rgb = np.asarray(Image.open(root / validation.cases[index]['input']).convert('RGB'))
                    rgb[j][~support[j]] = input_rgb[~support[j]]
                quantized = torch.from_numpy((rgb.astype(np.float32) / 255).transpose(0, 3, 1, 2)).cuda()
                embeddings = identity.embedding(quantized, batch['mask'], batch['grid']).cpu().numpy()
                input_embeddings = identity.embedding(batch['low'], batch['mask'], batch['grid']).cpu().numpy() if include_input else None
            for j, index in enumerate(batch['index'].tolist()):
                case = validation.cases[index]; ref = refs[case['reference_id']]
                target = np.asarray(Image.open(root / ref['target']).convert('RGB'))
                name = stage + '/images/' + case['id'] + '.png'; (out / name).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb[j]).save(out / name)
                embedfile = stage + '/embeddings/' + case['id'] + '.npy'; (out / embedfile).parent.mkdir(exist_ok=True)
                np.save(out / embedfile, embeddings[j], allow_pickle=False)
                row = {**case, **exported_pixel_metrics(rgb[j], target, support[j]), 'prediction': name,
                    'embedding': embedfile, 'ArcFace_observed_fixed': float(np.clip(embeddings[j] @ target_embeds[ref['id']], -1, 1))}
                if case['id'] in p['validation_preview_case_ids']:
                    file = stage + '/raw_float/' + case['id'] + '.npy'; (out / file).parent.mkdir(exist_ok=True)
                    np.save(out / file, raw[j], allow_pickle=False); row['raw_float'] = file
                rows.append(row)
                if include_input:
                    image = np.asarray(Image.open(root / case['input']).convert('RGB'))
                    file = stage + '/input_embeddings/' + case['id'] + '.npy'; (out / file).parent.mkdir(exist_ok=True)
                    np.save(out / file, input_embeddings[j], allow_pickle=False)
                    input_rows.append({**case, **exported_pixel_metrics(image, target, support[j]), 'embedding': file,
                        'ArcFace_observed_fixed': float(np.clip(input_embeddings[j] @ target_embeds[ref['id']], -1, 1))})
            if len(rows) % 100 == 0:
                print(f'{stage} validation {len(rows)}/520 elapsed={time.monotonic()-started:.0f}s', flush=True)
        grid = Image.new('RGB', (780, 2904), '#eeeeee'); draw = ImageDraw.Draw(grid)
        for j, label in enumerate(['input', 'DGP observed output', 'unchanged reference']):
            draw.text((2 + j * 260, 3), label, fill='black')
        by_id = {r['id']: r for r in rows}
        for i, cid in enumerate(p['validation_preview_case_ids']):
            row = by_id[cid]; y = 24 + i * 288
            for j, file in enumerate([root / row['input'], out / row['prediction'], root / refs[row['reference_id']]['target']]):
                draw.text((2 + j * 260, y + 2), cid, fill='black')
                with Image.open(file) as image:
                    grid.paste(image.convert('RGB'), (2 + j * 260, y + 28))
        name = stage + '/preview_10_rows.png'; grid.save(out / name)
        result = {'complete': True, 'rows': rows, 'summary': aggregate(rows), 'input_rows': input_rows,
            'input_summary': aggregate(input_rows) if include_input else None, 'preview': name,
            'seconds': time.monotonic() - stage_start, 'evaluation_basis': 'exact observed exported RGB PNG'}
        write(out / stage / 'metrics.json', result)
        return result

    baseline = evaluate('baseline', True)
    assert state_hash(model) == START_STATE and buffer_hash(model) == buffers
    save_state(out / 'checkpoints/baseline.pth')
    backbone = list(model.fpn.features.parameters()); backbone_ids = {id(x) for x in backbone}
    optimizer = torch.optim.Adam([{'params': backbone, 'lr': 2e-5},
        {'params': [x for x in model.parameters() if id(x) not in backbone_ids], 'lr': 1e-4}], weight_decay=1e-5)
    best = baseline['summary']; selected_epoch = 0; selected_checkpoint = 'checkpoints/baseline.pth'; snapshots = []
    training_started = time.monotonic()
    for epoch in range(1, 21):
        batches = [[train_indices[c] for c in batch] for batch in schedule[str(epoch)]]
        epoch_losses = []
        for step, batch in enumerate(DataLoader(fitting, batch_sampler=batches, num_workers=0), 1):
            clock(); model.train(); freeze_normalization(model); batch = device(batch)
            assert list(batch['case_id']) == schedule[str(epoch)][step - 1]
            optimizer.zero_grad(set_to_none=True)
            prediction = composite(model(batch['low']), batch['low'], batch['mask'])
            loss, values = objective(prediction, batch)
            assert torch.isfinite(loss) and torch.isfinite(values).all()
            loss.backward(); CONTEXT['backwards'] += 1
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
            assert norm > 0 and all(param.grad is None for param in identity.parameters())
            optimizer.step(); CONTEXT['updates'] += 1
            raw_values = values.detach().cpu().tolist(); epoch_losses.append(float(loss.detach()))
            trace = {'update': CONTEXT['updates'], 'epoch': epoch, 'step': step,
                'cases': list(batch['case_id']), 'loss': float(loss.detach()), 'raw_case_MSE': raw_values,
                'weights': [weights[group(train_by_id[c])] for c in batch['case_id']],
                'preclip_norm': norm, 'calibration_sha256': calibration_sha}
            with (out / 'updates.jsonl').open('a', encoding='utf-8') as stream:
                import json
                stream.write(json.dumps(trace, allow_nan=False) + '\n')
            if CONTEXT['updates'] == 32:
                measured = time.monotonic() - training_started
                projected = time.monotonic() - started + (7820 - 32) * measured / 32 + 4 * baseline['seconds'] + 60
                write(out / 'timing_at32.json', {'measured32_training_seconds': measured,
                    'baseline_evaluation_seconds': baseline['seconds'], 'projected_trainer_seconds': projected,
                    'trainer_cap_seconds': 2400, 'remaining_evaluations': 4, 'export_allowance_seconds': 60})
                if projected > 2400:
                    raise TimeoutError('Measured timing projection exceeds frozen2400-second trainer cap')
            if CONTEXT['updates'] == 1 or CONTEXT['updates'] % 100 == 0:
                print(f'mixed_v9 epoch{epoch}/20 update{CONTEXT["updates"]}/7820 loss={trace["loss"]:.6f} elapsed={time.monotonic()-started:.0f}s', flush=True)
        assert buffer_hash(model) == buffers
        print(f'epoch{epoch} complete mean_weighted_MSE={math.fsum(epoch_losses)/len(epoch_losses):.6f}', flush=True)
        if epoch in design['evaluation_epochs']:
            result = evaluate('epoch' + str(epoch))
            checkpoint = 'checkpoints/epoch_' + str(epoch) + '.pth'; save_state(out / checkpoint)
            qualifies_now = qualifies(result['summary'], baseline['summary'], best)
            if qualifies_now:
                selected_epoch = epoch; selected_checkpoint = checkpoint; best = result['summary']
            snapshots.append({'epoch': epoch, 'update': CONTEXT['updates'], 'checkpoint': checkpoint,
                'state_hash': state_hash(model), 'metrics': 'epoch' + str(epoch) + '/metrics.json',
                'qualifies': qualifies_now, 'selected_epoch_after': selected_epoch})
    assert CONTEXT['updates'] == CONTEXT['backwards'] == 7820
    assert state_hash(identity) == teacher_state and buffer_hash(model) == buffers
    verify(root, expected_protocol_sha)
    shutil.copyfile(out / selected_checkpoint, out / 'best.pth')
    write(out / 'selection.json', {'selected_epoch': selected_epoch, 'selected_source': selected_checkpoint,
        'best_sha256': sha(out / 'best.pth'), 'selection': design['selection'],
        'reason': 'Metric-eligible candidate; visual/native and independent final review still required' if selected_epoch else 'No trained candidate passed; retain starting V2 baseline',
        'production_promoted': False, 'native_visual_review_pending': True, 'independent_final_review_pending': True})
    clock()
    artifacts = {f.relative_to(out).as_posix(): sha(f) for f in sorted(out.rglob('*')) if f.is_file()}
    write(out / 'results.json', {'complete': True, 'version': 9, 'protocol_sha256': expected_protocol_sha,
        'total_optimizer_updates': 7820, 'training_backward_calls': 7820, 'preflight_autograd_calls': 1,
        'elapsed_seconds': time.monotonic() - started, 'snapshots': snapshots, 'selected_epoch': selected_epoch,
        'buffers_and_teacher_unchanged': True, 'calibration_sha256': calibration_sha, 'artifacts_sha256': artifacts,
        'validation_used': True, 'native_used': False, 'native_reserved_used': False,
        'production_promoted': False, 'model_improvement_established': False})
    print(f'V9 complete:7820 updates; selected_epoch={selected_epoch}; no production promotion', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--expected-protocol-sha', required=True)
    args = parser.parse_args()
    try:
        train(args.root.resolve(), args.expected_protocol_sha)
    except Exception as error:
        out = CONTEXT['out']
        if out is not None and out.exists() and not (out / 'failure.json').exists():
            sys.path.insert(0, str(args.root))
            from cctv_dgp_mixed_v9 import sha, write
            failure = {'complete': False, 'error_type': type(error).__name__, 'error': str(error),
                'optimizer_updates_recorded': CONTEXT['updates'], 'backward_calls_recorded': CONTEXT['backwards'],
                'preflight_autograd_calls_recorded': CONTEXT['autograd'], 'resume_permitted': False,
                'production_promoted': False}
            if CONTEXT['model'] is not None:
                try:
                    import torch
                    file = out / 'partial_state.pth'
                    torch.save({k: v.detach().cpu().clone() for k, v in CONTEXT['model'].state_dict().items()}, file)
                    failure['partial_state_sha256'] = sha(file)
                except Exception as export_error:
                    failure['partial_export_error'] = str(export_error)
            write(out / 'failure.json', failure)
        raise

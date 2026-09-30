"""Fixed 21-batch retention feasibility pilot. Training requires Linux CUDA VM."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path
import sys
import tarfile
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coverage_retention import guarded_step
from completion_inference import load_completion
from detector_training import load_manifest, ReviewedMasks, evaluate, detector_optimizer, segmentation_loss
from detector_replay import replay_consistency, retention_passes, BenchmarkMasks
from scripts.train_coverage_vm import require_vm_gpu, sha, safe_path, real_gate, mask_export, PROTOCOL_SHA, PARENT_SHA

INVENTORY_SHA = 'c831635933c9b615b4849ad23811a3b82ebde1883aa30a5c2249b8879fa95623'
REPLAY_SHA = 'ad6fd64aed4f773c968cac7748be0fd99a7c8679d1019e32fc64a03b13662dc7'


def replay_weights(protocol, cache):
    schedule = protocol['schedules']['extended']['batches']
    if len(schedule) != 10 or any(len(e) != 21 for e in schedule):
        raise ValueError('Unexpected schedule')
    weights = Counter(i - 73 for e in schedule for b in e for i in b if i >= 73)
    if set(weights) != set(cache) or len(weights) != 638:
        raise ValueError('Replay membership mismatch')
    totals = Counter()
    for i, weight in weights.items():
        positive = bool(cache[i][1].any())
        if positive != (i % 5 != 0):
            raise ValueError('Replay mask/type mismatch')
        totals['covered' if positive else 'clear'] += weight
    if totals != {'covered': 420, 'clear': 420}:
        raise ValueError('Replay exposure mismatch')
    for epoch in schedule:
        for batch in epoch:
            if len(batch) != 8 or sum(i < 73 for i in batch) != 4 or any(i < 0 for i in batch):
                raise ValueError('Mixed schedule mismatch')
    return weights


def next_rejection_streak(previous, accepted):
    return 0 if accepted else previous + 1


@torch.inference_mode()
def replay_losses(model, cache, weights, device):
    model.segmenter.eval()
    sums = {'covered': 0., 'clear': 0.}
    totals = {'covered': 0, 'clear': 0}
    ids = sorted(weights)
    for start in range(0, len(ids), 8):
        batch = ids[start:start + 8]
        x = torch.stack([cache[i][0].float() / 255 for i in batch]).to(device)
        m = torch.stack([cache[i][1].float() for i in batch]).to(device)
        logits = model.detect(x)
        for j, i in enumerate(batch):
            group = 'covered' if i % 5 else 'clear'
            loss = float(segmentation_loss(logits[j:j+1], m[j:j+1], .25, .1))
            sums[group] += weights[i] * loss
            totals[group] += weights[i]
    return {k: sums[k] / totals[k] for k in sums}


def main(args):
    require_vm_gpu()  # Must precede file creation, model load and optimizer.
    root = Path.cwd().resolve()
    out = safe_path(root, args.output)
    archive = root / 'retention-results.tar.gz'
    if out.exists() or (not args.preflight and archive.exists()):
        raise ValueError('Preserve existing outputs/archive; inspect before restarting')
    inventory = root / 'outputs/coverage_protocol_v1/vm_inventory.json'
    if sha(inventory) != INVENTORY_SHA:
        raise ValueError('Inventory changed')
    for path, digest in json.loads(inventory.read_text()).items():
        if sha(safe_path(root, path)) != digest:
            raise ValueError(f'Inventory mismatch: {path}')
    protocol_path = root / 'outputs/coverage_protocol_v1/protocol.json'
    if sha(protocol_path) != PROTOCOL_SHA:
        raise ValueError('Protocol changed')
    protocol = json.loads(protocol_path.read_text())
    for path, digest in protocol['input_hashes'].items():
        if sha(safe_path(root, path)) != digest:
            raise ValueError(f'Protocol input mismatch: {path}')
    cache_path = root / 'outputs/coverage_training_vm/replay_pixels.pth'
    if sha(cache_path) != REPLAY_SHA:
        raise ValueError('Replay cache changed')
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    weights = replay_weights(protocol, cache)
    rows = load_manifest(root / 'dataset/detector_training_extension_v2/manifest.json')
    real = ReviewedMasks([r for r in rows if r['split'] == 'train'], 256)
    if len(real) != 73:
        raise ValueError('Real training split changed')
    parent = root / 'outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth'
    if sha(parent) != PARENT_SHA:
        raise ValueError('Parent checkpoint changed')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, state = load_completion(parent, 'cuda')
    model.eval()
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    teacher = copy.deepcopy(model.segmenter).eval().requires_grad_(False)
    schedule = protocol['schedules']['extended']['batches'][0]

    def batch_tensors(batch):
        items = [real[i] if i < 73 else (cache[i-73][0].float()/255, cache[i-73][1].float()) for i in batch]
        x, m = (torch.stack([item[j] for item in items]).cuda() for j in (0, 1))
        tag = torch.tensor([i >= 73 for i in batch], device='cuda')
        return x, m, tag

    out.mkdir(parents=True)
    started = time.monotonic()
    ceilings = replay_losses(model, cache, weights, 'cuda')
    if any(not torch.isfinite(torch.tensor(v)) for v in ceilings.values()):
        raise ValueError('Nonfinite parent ceiling')
    metadata = dict(parent_sha256=PARENT_SHA, protocol_sha256=PROTOCOL_SHA,
                    inventory_sha256=INVENTORY_SHA, replay_sha256=REPLAY_SHA,
                    script_sha256=sha(__file__), helper_sha256=sha(root/'coverage_retention.py'),
                    specification_sha256=sha(root/'COVERAGE_RETENTION_PROTOCOL.md'),
                    torch=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                    ceilings=ceilings, tolerance=1e-6, factors=[1., .5, .25, .125],
                    lr=1e-5, weight_decay=1e-4, background_weight=.25, consistency_weight=1.,
                    clip=1., max_scheduled_batches=21, stop_rejection_streak=3,
                    preflight=args.preflight)
    (out/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    if args.preflight:
        x, m, _ = batch_tensors(schedule[0])
        with torch.no_grad():
            loss = segmentation_loss(model.detect(x), m, .25, .1)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite preflight loss')
        (out/'preflight.json').write_text(json.dumps({'optimizer_updates': 0, 'finite_loss': float(loss), 'ceilings': ceilings})+'\n')
        print('CUDA preflight complete; zero updates', flush=True)
        return

    val_rows = [r for r in rows if r['split'] == 'validation']
    datasets = {'real': ReviewedMasks(val_rows, 256), 'synthetic': BenchmarkMasks(root/'outputs/downloaded_completion_benchmark/outputs/completion_pretrained_vm', 256)}
    for name, predicate in [('human_real', lambda r: Path(r['image']).name != 'new_covered_40.png'),
                            ('mannequin', lambda r: Path(r['image']).name == 'new_covered_40.png'),
                            ('glare', lambda r: r.get('glare_stratum') == 'strong_lens_reflection')]:
        subset = [r for r in val_rows if predicate(r)]
        if subset:
            datasets[name] = ReviewedMasks(subset, 256)
    loaders = {k: torch.utils.data.DataLoader(v, batch_size=8) for k, v in datasets.items()}
    baseline = {k: evaluate(model, v, 'cuda') for k, v in loaders.items()}
    (out/'baseline.json').write_text(json.dumps(baseline, indent=2)+'\n')
    torch.save(initial, out/'initial.pth')
    optimizer = detector_optimizer(model, 1e-5)
    accepted = streak = trials = 0
    reason = 'scheduled_budget_complete'
    for step, batch in enumerate(schedule, 1):
        model.segmenter.train()
        x, m, tag = batch_tensors(batch)
        optimizer.zero_grad(set_to_none=True)
        logits = model.detect(x)
        with torch.no_grad():
            reference = teacher(x[tag])
        loss = segmentation_loss(logits, m, .25, .1) + replay_consistency(logits[tag], reference, torch.ones(int(tag.sum()), dtype=torch.bool, device='cuda'))
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite mixed loss')
        loss.backward()
        pre_clip = float(torch.nn.utils.clip_grad_norm_(model.segmenter.parameters(), 1., error_if_nonfinite=True))
        result = guarded_step(model.segmenter, optimizer, lambda: replay_losses(model, cache, weights, 'cuda'), ceilings)
        accepted += int(result['accepted'])
        trials += len(result['attempts'])
        streak = next_rejection_streak(streak, result['accepted'])
        result.update(step=step, indices=batch, accepted_updates=accepted, rejection_streak=streak,
                      mixed_loss=float(loss.detach()), pre_clip_norm=pre_clip, elapsed_seconds=time.monotonic()-started)
        with (out/'steps.jsonl').open('a') as f:
            f.write(json.dumps(result)+'\n')
        print(json.dumps(result), flush=True)
        if streak >= 3:
            reason = 'three_consecutive_rejected_batches'
            break
    model.eval()
    if not all(torch.equal(v.cpu(), initial['generator.'+k]) for k, v in model.generator.state_dict().items()):
        raise ValueError('Generator changed')
    final_replay = replay_losses(model, cache, weights, 'cuda')
    if not all(final_replay[k] <= ceilings[k]+1e-6 for k in ceilings):
        raise ValueError('Retained state violates replay constraint')
    metrics = {k: evaluate(model, v, 'cuda') for k, v in loaders.items()}
    r_ok = real_gate(metrics['real'], baseline['real'], baseline['real']['iou'])
    s_ok = retention_passes(metrics['synthetic'], baseline['synthetic'])
    selected = r_ok and s_ok
    report = dict(complete=True, scheduled_batches=step, accepted_updates=accepted, trials=trials,
                  stop_reason=reason, selected=selected, real_gate=r_ok, synthetic_gate=s_ok,
                  final_replay=final_replay, elapsed_seconds=time.monotonic()-started, **metrics)
    (out/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    export = {**state, 'model': model.state_dict(), 'detector_only': True, 'detector_training': metadata,
              'retention_pilot': report}
    torch.save(export, out/'final.pth')
    if selected:
        torch.save(export, out/'best_detector.pth')
    for domain in ('real', 'synthetic'):
        mask_export(model, datasets[domain], out/'final_masks'/domain)
    (out/'complete.json').write_text(json.dumps({'complete': True, 'final_sha256': sha(out/'final.pth'), 'promotion_requires_visual_review': True})+'\n')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(out, arcname=out.relative_to(root).as_posix())
        for path in ('scripts/train_retention_vm.py', 'coverage_retention.py', 'COVERAGE_RETENTION_PROTOCOL.md'):
            tar.add(root/path, arcname=path)
    print('Download', archive, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/retention_training_vm')
    parser.add_argument('--preflight', action='store_true')
    main(parser.parse_args())

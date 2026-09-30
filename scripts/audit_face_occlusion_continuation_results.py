"""Verify the fixed VM continuation return; local inference only, no optimizer."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_face_occlusion_results import (
    canonical_member, read_json, read_lines, require, same_scores, subsets,
    verified_inputs, verify_bundle,
)
from scripts.package_face_occlusion_vm import sha, sha_stream
from scripts.train_coverage_vm import real_gate, PROTOCOL_SHA, PARENT_SHA
from detector_replay import retention_passes
from scripts.evaluate_coverage_results import binary, recount

START_SHA = 'cdd1752bce8e4a087ce2aac5af73ba81b6316ac9118e47f12acd7a24fcb62669'
BUNDLE_SHA = '93bab8a34d23a48cd8f9232dcbe1cd65c91bd024b2cd6e8480d76e863f82e3f7'
INVENTORY_SHA = '5d155a4373f8a251ecd9d0bb6caa91cf7f0b549d96bdea0b13f7760fbb2c1888'
FACE_INVENTORY_SHA = 'cb8485ef260bf83f8f4f3e6ce2f6fd45dc1a8726980130ddf4dfea956e83a4c1'
OLD_INVENTORY_SHA = 'c831635933c9b615b4849ad23811a3b82ebde1883aa30a5c2249b8879fa95623'
REPLAY_SHA = 'ad6fd64aed4f773c968cac7748be0fd99a7c8679d1019e32fc64a03b13662dc7'
INVENTORY_PATH = 'outputs/face_occlusion_continuation_bundle_v1/inventory.json'
PREFIX = 'outputs/face_occlusion_continuation_vm'
ADDITIONAL_EPOCHS = (1, 5, 10, 20)
SIZES = {'real': 25, 'synthetic': 400, 'training_real': 73}
DOMAINS = (*SIZES, 'human_real', 'mannequin', 'glare')
EXTRACTION = ROOT / 'outputs/downloaded_face_occlusion_continuation'
AUDIT_OUTPUT = ROOT / 'outputs/face_occlusion_continuation_validation'
ARCHIVE = ROOT / 'outputs/face-occlusion-continuation-results.tar.gz'


def audit_history(steps, history, schedule, baseline, complete):
    require(complete.get('complete') is True and complete.get('parent_unchanged') is True,
            'Missing completion/invariance marker')
    require(complete.get('source_sha256') == START_SHA, 'Source checkpoint differs')
    require(len(schedule) == 10 and all(len(e) == 21 for e in schedule), 'Original schedule differs')
    require(len(steps) == 420, 'Require exactly 420 additional logged steps')
    original = [batch for epoch in schedule for batch in epoch]
    for n, (row, batch) in enumerate(zip(steps, original + original), 1):
        expected = {'additional_epoch': (n - 1) // 21 + 1, 'global_epoch': (n - 1) // 21 + 11,
                    'step': (n - 1) % 21 + 1, 'fresh_optimizer_updates': n,
                    'cumulative_optimizer_updates': 210 + n}
        require(all(type(row.get(k)) is int and row[k] == v for k, v in expected.items()),
                'Executed update counter differs')
        require(row.get('indices') == batch, 'Executed schedule differs')
        require(all(type(row.get(k)) in (int, float) and math.isfinite(row[k]) and row[k] >= 0
                    for k in ('loss', 'pre_clip_norm')), 'Loss/gradient must be finite and nonnegative')
    require([r.get('additional_epoch') for r in history] == list(ADDITIONAL_EPOCHS),
            'Candidate check epochs differ')
    best = baseline['real']['iou']; selected = []
    for row, e in zip(history, ADDITIONAL_EPOCHS):
        counters = {'epoch': e + 10, 'optimizer_updates': 210 + e * 21,
                    'additional_epoch': e, 'fresh_optimizer_updates': e * 21}
        require(all(type(row.get(k)) is int and row[k] == v for k, v in counters.items()),
                'Candidate update counter differs')
        mean = sum(r['loss'] for r in steps[(e - 1) * 21:e * 21]) / 21
        require(type(row.get('mean_epoch_loss')) in (int, float) and
                math.isclose(mean, row['mean_epoch_loss'], rel_tol=0, abs_tol=1e-12),
                'Epoch loss differs from step log')
        real = real_gate(row['real'], baseline['real'], best)
        synthetic = retention_passes(row['synthetic'], baseline['synthetic'])
        require(all(type(row.get(k)) is bool for k in ('real_gate', 'synthetic_gate', 'selected')) and
                (row['real_gate'], row['synthetic_gate'], row['selected']) == (real, synthetic, real and synthetic),
                'Logged selection differs from unchanged gates')
        if real and synthetic:
            best = row['real']['iou']; selected.append(e + 10)
    require(type(complete.get('fresh_optimizer_updates')) is int and
            type(complete.get('cumulative_model_updates')) is int and
            (complete['fresh_optimizer_updates'], complete['cumulative_model_updates']) == (420, 630),
            'Completion update counters differ')
    require(complete.get('selected_epochs') == selected, 'Completion selection differs')
    return {'fresh_optimizer_updates': 420, 'cumulative_model_updates': 630, 'selected_epochs': selected}


def expected_files(inventory, *, selected):
    files = set(inventory) | {INVENTORY_PATH}
    files.update(f'{PREFIX}/{name}' for name in (
        'run.json', 'complete.json', 'baseline.json', 'source_metrics.json',
        'steps.jsonl', 'metrics.jsonl', 'initial.pth', 'final_optimizer.pth'))
    for folder, domains in [('baseline_masks', ('real', 'synthetic')), ('initial_masks', SIZES)]:
        files.update(f'{PREFIX}/{folder}/{d}/{i:04}.png' for d in domains for i in range(SIZES[d]))
    for e in ADDITIONAL_EPOCHS:
        files.add(f'{PREFIX}/epoch_{e + 10}.pth')
        files.update(f'{PREFIX}/epoch_{e + 10}_masks/{d}/{i:04}.png'
                     for d in SIZES for i in range(SIZES[d]))
    if selected: files.add(f'{PREFIX}/best_detector.pth')
    return files


def member_limit(name):
    canonical_member(name)
    if name == f'{PREFIX}/final_optimizer.pth': return 128 * 1024 * 1024
    if name.endswith('.pth'): return 64 * 1024 * 1024
    if name.endswith('.png'): return 512 * 1024
    return 2 * 1024 * 1024


def verify_package():
    old = verify_bundle(ROOT / 'outputs/face-occlusion-vm-code.tar.gz',
                        ROOT / 'outputs/face_occlusion_bundle_v1/inventory.json')
    inventory_path = ROOT / INVENTORY_PATH
    bundle = ROOT / 'outputs/face-occlusion-continuation-code.tar.gz'
    require(sha(bundle) == BUNDLE_SHA and sha(inventory_path) == INVENTORY_SHA,
            'Continuation package/inventory changed')
    inventory = read_json(inventory_path)
    expected = {**inventory, INVENTORY_PATH: INVENTORY_SHA}; seen = set()
    with tarfile.open(bundle, 'r|gz') as tar:
        for member in tar:
            name = canonical_member(member.name)
            require(member.isfile() and name not in seen and name in expected,
                    'Unexpected sent package member')
            require(sha_stream(tar.extractfile(member)) == expected[name], 'Sent package member changed')
            seen.add(name)
    require(seen == set(expected), 'Sent package file missing')
    for path, digest in inventory.items():
        canonical_member(path)
        require(sha(ROOT / path) == digest, f'Local continuation reference changed: {path}')
    return inventory, old


def extract_return(archive, output, inventory):
    require(output.resolve().is_relative_to(ROOT) and not output.exists(), 'Preserve existing extracted evidence')
    allowed = expected_files(inventory, selected=True)
    allowed_dirs = {str(parent.as_posix()) for name in allowed for parent in Path(name).parents
                    if parent.as_posix() == PREFIX or parent.as_posix().startswith(PREFIX + '/')}
    provenance = {**inventory, INVENTORY_PATH: INVENTORY_SHA}
    output.mkdir(parents=True); seen = set(); hashes = {}; total = 0
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            name = canonical_member(member.name)
            require(name not in seen and len(seen) < 3200 and (member.isfile() or member.isdir()),
                    'Duplicate path, excessive members or unsupported archive type')
            seen.add(name)
            if member.isdir():
                require(name in allowed_dirs, 'Unexpected returned directory')
                continue
            require(name in allowed, 'Unexpected returned file')
            total += member.size
            require(0 <= member.size <= member_limit(name) and total <= 600 * 1024 * 1024,
                    'Returned file/archive exceeds fixed size bound')
            destination = (output / name).resolve()
            require(destination.is_relative_to(output.resolve()), 'Extraction escapes destination')
            destination.parent.mkdir(parents=True, exist_ok=True); digest = hashlib.sha256()
            with tar.extractfile(member) as stream, destination.open('xb') as saved:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk); saved.write(chunk)
            hashes[name] = digest.hexdigest()
            if name in provenance:
                require(hashes[name] == provenance[name], f'Returned provenance differs: {name}')
    require(set(provenance).issubset(hashes), 'Returned provenance missing')
    return hashes


def checkpoint_audit(payload, initial, run, report):
    import torch
    expected = {k: v for k, v in initial.items() if k != 'model'}
    expected.update({k: report[k] for k in ('epoch', 'optimizer_updates', 'additional_epoch', 'fresh_optimizer_updates')})
    expected.update(continuation_metadata=run, selection=report)
    require(set(payload) == set(expected) | {'model'} and
            all(payload[k] == v for k, v in expected.items()), 'Checkpoint metadata differs')
    state = payload['model']; before = initial['model']
    require(set(state) == set(before) and not any(k.startswith(('generator.', 'segmenter.')) for k in state),
            'Unexpected detector state membership')
    require(all(isinstance(v, torch.Tensor) and v.shape == before[k].shape and v.dtype == before[k].dtype
                and torch.isfinite(v).all() for k, v in state.items()), 'Invalid checkpoint tensor')
    frozen = [k for k in state if k.startswith('reference_visible_head.') or
              k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))]
    require(frozen and all(torch.equal(state[k], before[k]) for k in frozen), 'Frozen source state changed')
    changes = {prefix: sum(not torch.equal(v, before[k]) for k, v in state.items() if k.startswith(prefix))
               for prefix in ('network.encoder.', 'network.decoder.', 'network.segmentation_head.')}
    require(all(changes.values()), 'No changes in a declared trainable component')
    return {'frozen_tensors_verified': len(frozen), 'changed_tensors': changes}


def audit_optimizer(payload, groups, model_sha):
    import torch
    require(set(payload) == {'state', 'model_sha256', 'fresh_optimizer_updates', 'cumulative_model_updates'} and
            type(payload['fresh_optimizer_updates']) is int and type(payload['cumulative_model_updates']) is int and
            payload['model_sha256'] == model_sha and payload['fresh_optimizer_updates'] == 420 and
            payload['cumulative_model_updates'] == 630, 'Final optimizer model binding/counters differ')
    saved = payload['state']
    require(set(saved) == {'state', 'param_groups'} and len(saved['param_groups']) == len(groups) == 2,
            'Optimizer groups differ')
    ids = []; parameters = []
    for stored, expected in zip(saved['param_groups'], groups):
        allowed_keys = {'params', 'lr', 'weight_decay', 'betas', 'eps', 'amsgrad', 'maximize',
                        'capturable', 'differentiable', 'foreach', 'fused', 'decoupled_weight_decay'}
        require(set(stored).issubset(allowed_keys) and
                ('decoupled_weight_decay' not in stored or stored['decoupled_weight_decay'] is True),
                'Unexpected optimizer setting or coupled decay')
        require(stored.get('lr') == expected['lr'] and stored.get('weight_decay') == 1e-4 and
                tuple(stored.get('betas', ())) == (.9, .999) and stored.get('eps') == 1e-8,
                'Optimizer rate/decay/moments policy differs')
        require(all(stored.get(k) is False for k in ('amsgrad', 'maximize', 'capturable', 'differentiable')) and
                stored.get('foreach') is None and stored.get('fused') is None,
                'Optimizer execution flags differ')
        require(isinstance(stored.get('params'), list) and len(stored['params']) == len(expected['params']),
                'Optimizer group parameter count differs')
        ids.extend(stored['params']); parameters.extend(expected['params'])
    require(all(type(i) is int for i in ids) and ids == list(range(len(parameters))) and
            set(saved['state']) == set(ids), 'Optimizer parameter IDs duplicate/missing/out of order')
    for i, parameter in zip(ids, parameters):
        state = saved['state'][i]
        require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Optimizer moment fields differ')
        step = state['step']
        require(isinstance(step, torch.Tensor) and step.numel() == 1 and
                torch.isfinite(step).all() and step.item() == 420, 'Optimizer fresh step count differs')
        require(all(isinstance(state[k], torch.Tensor) and state[k].shape == parameter.shape and
                    state[k].dtype == parameter.dtype and torch.isfinite(state[k]).all()
                    for k in ('exp_avg', 'exp_avg_sq')), 'Optimizer moment shape/dtype/finiteness differs')
        require((state['exp_avg_sq'] >= 0).all(), 'Optimizer variance moment must be nonnegative')
    return {'parameter_states_verified': len(ids), 'parameter_elements': sum(p.numel() for p in parameters),
            'step_per_parameter': 420, 'model_sha256': model_sha, 'optimizer_updates_locally': 0}


def validate_run(run, inventory, old):
    setup_path = ROOT / 'outputs/downloaded_face_occlusion/outputs/face_occlusion_dependencies/setup.json'
    setup_sha = '6af41c0662a0a91d55871c4faadb6d0810d3e366a964a4da23c5b52183ef210a'
    require(sha(setup_path) == setup_sha, 'Previously audited VM setup changed')
    setup = read_json(setup_path)
    fixed = {'preflight': False, 'source_checkpoint_sha256': START_SHA, 'source_epoch': 10,
             'source_optimizer_updates': 210, 'optimizer_reset': True, 'optimizer_state_restored': False,
             'additional_epochs': 20, 'additional_updates': 420,
             'check_additional_epochs': list(ADDITIONAL_EPOCHS), 'protocol_sha256': PROTOCOL_SHA,
             'replay_sha256': REPLAY_SHA, 'parent_sha256': PARENT_SHA,
             'old_inventory_sha256': OLD_INVENTORY_SHA, 'face_inventory_sha256': FACE_INVENTORY_SHA,
             'continuation_inventory_sha256': INVENTORY_SHA, 'encoder_lr': 1e-5,
             'decoder_head_lr': 1e-4, 'weight_decay': 1e-4, 'clip': 1., 'seed': 42,
             'background_weight': .25, 'consistency_weight': 1., 'validation_batch_size': 1,
             'batchnorm_policy': 'fixed_running_statistics_trainable_affine',
             'script_sha256': inventory['scripts/train_face_occlusion_continuation_vm.py'],
             'specification_sha256': inventory['FACE_OCCLUSION_CONTINUATION.md'],
             'adapter_sha256': old['face_occlusion_adapter.py'], 'setup_sha256': setup_sha,
             'dependencies': setup['packages'], 'torch': setup['torch'],
             'test_split': 'Not evaluated; previous development work inspected it',
             'overlap_limitation': 'External FaceExtraction FFHQ pretraining overlap unresolved'}
    for key, value in fixed.items():
        require(key in run and type(run[key]) is type(value) and run[key] == value,
                f'Fixed run field differs: {key}')
    require(set(run) == set(fixed) | {'python', 'gpu'} and
            all(isinstance(run[k], str) and run[k] for k in ('python', 'gpu')), 'Run metadata fields differ')


def audit():
    import numpy as np
    from PIL import Image, ImageDraw
    import torch
    require(ARCHIVE.is_file(), f'VM return archive is absent: {ARCHIVE}')
    require(not AUDIT_OUTPUT.exists(), 'Preserve existing audit evidence')
    inventory, old = verify_package()
    protocol, rows, data = verified_inputs()
    require({k: len(v) for k, v in data.items()} == SIZES, 'Evaluation dataset sizes differ')
    hashes = extract_return(ARCHIVE, EXTRACTION, inventory); returned = EXTRACTION / PREFIX
    run = read_json(returned / 'run.json'); complete = read_json(returned / 'complete.json')
    baseline = read_json(returned / 'baseline.json'); source = read_json(returned / 'source_metrics.json')
    validate_run(run, inventory, old)
    require(set(baseline) == set(source) == set(DOMAINS), 'Baseline/source metric domains differ')
    steps = read_lines(returned / 'steps.jsonl'); history = read_lines(returned / 'metrics.jsonl')
    log = audit_history(steps, history, protocol['schedules']['extended']['batches'], baseline, complete)
    require(set(hashes) == expected_files(inventory, selected=bool(log['selected_epochs'])),
            'Returned file membership differs')
    require(hashes[f'{PREFIX}/initial.pth'] == START_SHA and
            hashes[f'{PREFIX}/epoch_30.pth'] == complete['final_sha256'] and
            hashes[f'{PREFIX}/final_optimizer.pth'] == complete['final_optimizer_sha256'],
            'Source/final/optimizer file hash differs')
    require(complete.get('promotion_requires_visual_review') is True and
            math.isfinite(complete['seconds']) and complete['seconds'] > 0 and
            0 < complete['peak_cuda_allocated_bytes'] <= complete['peak_cuda_reserved_bytes'],
            'Completion runtime/review marker differs')
    initial = torch.load(returned / 'initial.pth', map_location='cpu', weights_only=True)
    require(initial['initialization'] == 'pretrained' and initial['epoch'] == 10 and
            initial['optimizer_updates'] == 210, 'Source checkpoint metadata differs')
    same_scores(source, initial['selection'], 'source checkpoint')
    targets = {domain: [dataset[i][1][0].numpy().astype(bool) for i in range(len(dataset))]
               for domain, dataset in data.items()}
    def saved_scores(folder, domains):
        predictions = {d: [binary(folder / d / f'{i:04}.png') for i in range(SIZES[d])] for d in domains}
        scores = {d: recount(predictions[d], targets[d]) for d in domains}
        return subsets(scores, predictions['real'], targets['real'], rows), predictions
    base_scores, base_predictions = saved_scores(returned / 'baseline_masks', ('real', 'synthetic'))
    same_scores(base_scores, baseline, 'baseline')
    source_scores, source_predictions = saved_scores(returned / 'initial_masks', SIZES)
    same_scores(source_scores, source, 'source'); total = 425 + 498; candidates = []; previews = {}
    for row in history:
        e = row['epoch']; scores, predictions = saved_scores(returned / f'epoch_{e}_masks', SIZES)
        same_scores(scores, row, f'epoch {e}'); total += 498; previews[e] = predictions['real']
        candidates.append({**{k: row[k] for k in ('epoch', 'additional_epoch', 'optimizer_updates',
                                                 'fresh_optimizer_updates', 'real_gate', 'synthetic_gate', 'selected')}, **scores})
        print(f'Recounted epoch {e}: real IoU={scores["real"]["iou"]:.5f}, synthetic IoU={scores["synthetic"]["iou"]:.5f}', flush=True)
    require(total == 2915, 'Incomplete saved-mask recount')
    special = [i for i, row in enumerate(rows) if row.get('glare_stratum') == 'strong_lens_reflection' or
               Path(row['image']).name == 'new_covered_40.png']
    require(len(special) == 2, 'Glare/mannequin preview membership differs')
    covered = [i for i in range(25) if i not in special and targets['real'][i].any()][:4]
    clear = [i for i in range(25) if i not in special and not targets['real'][i].any()][:4]
    preview_ids = covered + clear + special
    require(len(set(preview_ids)) == 10, 'Ten-row preview membership differs')
    AUDIT_OUTPUT.mkdir(); sheet = Image.new('RGB', (1024, 1480), 'white')
    for line, i in enumerate(preview_ids):
        x, _ = data['real'][i]
        masks = [targets['real'][i], base_predictions['real'][i], source_predictions['real'][i]]
        masks.extend(previews[e][i] for e in (11, 15, 20, 30))
        tiles = [Image.fromarray(np.rint(x.permute(1, 2, 0).numpy() * 255).astype('uint8'))]
        tiles.extend(Image.fromarray(m.astype('uint8') * 255).convert('RGB') for m in masks)
        ImageDraw.Draw(sheet).text((2, line * 148 + 2),
                                  f'Validation {i}: input | target | parent | source10 | epoch11 | epoch15 | epoch20 | epoch30', fill='black')
        for j, tile in enumerate(tiles): sheet.paste(tile.resize((128, 128)), (j * 128, line * 148 + 20))
    sheet.save(AUDIT_OUTPUT / 'preview.png')
    evidence = {'archive_sha256': sha(ARCHIVE), 'bundle_sha256': BUNDLE_SHA, 'log_audit': log,
                'saved_masks_recounted': total, 'baseline': base_scores, 'source': source_scores,
                'candidates': candidates, 'preview_validation_indices': preview_ids, 'runtime': complete,
                'optimizer_updates_locally': 0, 'promoted': False,
                'limitations': ['Checkpoint-to-mask and final optimizer verification pending',
                    'Remote parent invariance is an executed-code check/log claim; parent tensors were not returned',
                    'Intermediate/best optimizer snapshots were not archived',
                    'External FFHQ pretraining overlap unresolved; reused development evidence only']}
    (AUDIT_OUTPUT / 'members.json').write_text(json.dumps(hashes, indent=2) + '\n')
    (AUDIT_OUTPUT / 'results.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({'saved_masks_recounted': total, **log, 'promoted': False}), flush=True)


def reproduce():
    import numpy as np
    import torch
    from completion_inference import load_completion
    from face_occlusion_adapter import load_adapter, parameter_groups
    from scripts.train_face_occlusion_vm import dependency_versions
    require((AUDIT_OUTPUT / 'results.json').is_file(), 'Run archive/log/mask audit first')
    destination = AUDIT_OUTPUT / 'reproduction.json'
    require(not destination.exists(), 'Preserve completed reproduction')
    audit_report = read_json(AUDIT_OUTPUT / 'results.json')
    require(sha(ARCHIVE) == audit_report['archive_sha256'], 'Return archive changed')
    inventory, old = verify_package(); _, rows, data = verified_inputs()
    hashes = read_json(AUDIT_OUTPUT / 'members.json')
    for path, digest in hashes.items():
        canonical_member(path)
        require(sha(EXTRACTION / path) == digest, f'Extracted evidence changed: {path}')
    returned = EXTRACTION / PREFIX; run = read_json(returned / 'run.json'); validate_run(run, inventory, old)
    baseline = read_json(returned / 'baseline.json'); source = read_json(returned / 'source_metrics.json')
    history = read_lines(returned / 'metrics.jsonl')
    torch.set_num_threads(4)
    sys.path.insert(0, str(ROOT / 'outputs/face_extraction_dependencies'))
    report = {'archive_sha256': audit_report['archive_sha256'], 'parent_sha256': PARENT_SHA,
              'torch': str(torch.__version__), 'dependencies': dependency_versions(ROOT / 'outputs/face_extraction_dependencies'),
              'device': 'cpu', 'optimizer_updates_locally': 0, 'promoted': False,
              'checkpoint_checks': {}, 'reproductions': {}}
    def infer(model, folder, logged, label, saved_domains):
        model.requires_grad_(False).eval()
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        scores = {}; differences = {}; real_predictions = real_truth = None
        with torch.inference_mode():
            for domain, dataset in data.items():
                predictions = []; truth = []; nonexact = []
                for i, (x, m) in enumerate(dataset):
                    p = (model.detect(x[None]).sigmoid()[0, 0] >= .5).numpy(); t = m[0].numpy().astype(bool)
                    predictions.append(p); truth.append(t)
                    if domain in saved_domains:
                        pixels = int(np.count_nonzero(p != binary(folder / domain / f'{i:04}.png')))
                        if pixels: nonexact.append({'index': i, 'different_pixels': pixels})
                    if (i + 1) % 100 == 0: print(label, domain, i + 1, len(dataset), flush=True)
                scores[domain] = recount(predictions, truth)
                if domain in saved_domains:
                    differences[domain] = {'cases': len(dataset), 'all_exact': not nonexact,
                                           'different_pixels': sum(r['different_pixels'] for r in nonexact),
                                           'nonexact_cases': nonexact}
                if domain == 'real': real_predictions, real_truth = predictions, truth
        subsets(scores, real_predictions, real_truth, rows)
        require(all(torch.equal(v.cpu(), before[k]) for k, v in model.state_dict().items()),
                'Read-only inference changed model state')
        deltas = {d: {k: float(v - logged[d][k]) for k, v in metrics.items()} for d, metrics in scores.items()}
        return {'scores': scores, 'mask_reproduction': differences,
                'metric_deltas_cpu_minus_logged': deltas, 'model_state_unchanged': True}
    parent_path = ROOT / 'outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth'
    require(sha(parent_path) == PARENT_SHA, 'Original parent/generator changed')
    parent, _ = load_completion(parent_path, 'cpu')
    report['reproductions']['parent'] = infer(parent, returned / 'baseline_masks', baseline, 'parent', ('real', 'synthetic'))
    del parent
    require(sha(returned / 'initial.pth') == START_SHA, 'Copied source changed')
    model, initial = load_adapter(returned / 'initial.pth', 'cpu')
    require(initial['initialization'] == 'pretrained' and initial['epoch'] == 10 and initial['optimizer_updates'] == 210,
            'Copied source metadata differs')
    report['reproductions']['source10'] = infer(model, returned / 'initial_masks', source, 'source10', SIZES)
    del model
    for logged in history:
        e = logged['epoch']; path = returned / f'epoch_{e}.pth'; model, payload = load_adapter(path, 'cpu')
        report['checkpoint_checks'][str(e)] = {'sha256': sha(path), **checkpoint_audit(payload, initial, run, logged)}
        if e == 30:
            final_sha = hashes[f'{PREFIX}/epoch_30.pth']
            optimizer = torch.load(returned / 'final_optimizer.pth', map_location='cpu', weights_only=True)
            report['final_optimizer'] = audit_optimizer(optimizer, parameter_groups(model), final_sha)
            del optimizer
        report['reproductions'][str(e)] = infer(model, returned / f'epoch_{e}_masks', logged, f'epoch{e}', SIZES)
        del model, payload
        (AUDIT_OUTPUT / 'reproduction_partial.json').write_text(json.dumps(report, indent=2) + '\n')
    selected = audit_report['log_audit']['selected_epochs']
    if selected:
        e = selected[-1]; logged = next(r for r in history if r['epoch'] == e)
        best = torch.load(returned / 'best_detector.pth', map_location='cpu', weights_only=True)
        candidate = torch.load(returned / f'epoch_{e}.pth', map_location='cpu', weights_only=True)
        checkpoint_audit(best, initial, run, logged)
        require(all(torch.equal(v, candidate['model'][k]) for k, v in best['model'].items()),
                'Best detector state differs from last eligible candidate')
        report['best_detector'] = {'epoch': e, 'sha256': hashes[f'{PREFIX}/best_detector.pth'], 'state_matches_candidate': True}
        del best, candidate
    cpu_base = report['reproductions']['parent']['scores']; best_iou = cpu_base['real']['iou']
    cpu_selection = []; agreed = True
    for logged in history:
        e = logged['epoch']; scores = report['reproductions'][str(e)]['scores']
        real = real_gate(scores['real'], cpu_base['real'], best_iou)
        synthetic = retention_passes(scores['synthetic'], cpu_base['synthetic']); eligible = real and synthetic
        cpu_selection.append({'epoch': e, 'real_gate': real, 'synthetic_gate': synthetic, 'selected': eligible})
        agreed &= (real, synthetic, eligible) == (logged['real_gate'], logged['synthetic_gate'], logged['selected'])
        if eligible: best_iou = scores['real']['iou']
    report['cpu_selection'] = cpu_selection; report['selection_agrees_with_vm'] = agreed
    report['saved_masks_compared'] = sum(d['cases'] for r in report['reproductions'].values() for d in r['mask_reproduction'].values())
    report['all_saved_masks_exact'] = all(d['all_exact'] for r in report['reproductions'].values() for d in r['mask_reproduction'].values())
    report['different_pixels'] = sum(d['different_pixels'] for r in report['reproductions'].values() for d in r['mask_reproduction'].values())
    require(report['saved_masks_compared'] == 2915, 'Incomplete checkpoint-to-mask reproduction')
    report['complete'] = True
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('saved_masks_compared', 'all_saved_masks_exact', 'different_pixels',
                                           'selection_agrees_with_vm', 'optimizer_updates_locally', 'promoted')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--reproduce', action='store_true', help='Verify checkpoint states/moments and reproduce all saved masks on CPU')
    mode.add_argument('--verify-package', action='store_true', help='Read-only check of the already sent package and local references')
    args = parser.parse_args()
    if args.verify_package:
        inventory, _ = verify_package()
        print(json.dumps({'bundle_sha256': BUNDLE_SHA, 'continuation_members_verified': len(inventory) + 1,
                          'optimizer_updates_locally': 0, 'vm_return_present': ARCHIVE.is_file()}))
    elif args.reproduce: reproduce()
    else: audit()

"""Independent reflection-pilot return audit; no model fitting or promotion."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_face_occlusion_results import (
    canonical_member, read_json, read_lines, require, same_scores, subsets, verified_inputs,
)
from scripts.package_face_occlusion_vm import sha, sha_stream
from scripts.train_coverage_vm import real_gate, PROTOCOL_SHA, PARENT_SHA
from scripts.train_reflection_coverage_vm import (
    DATA_PATH, DATA_SHA, PIXELS_SHA, AUDIT_SHA, INVENTORY_PATH,
    MODEL_SHA, OPTIMIZER_SHA, PRIOR_INVENTORIES,
)
from scripts.evaluate_coverage_results import binary, recount
from detector_replay import retention_passes
from face_occlusion_focus import validate_source, validate_optimizer

BUNDLE_SHA = '4574929a1c54729ad078f4456dc4bf44cda65cc207ef28c1f83b5e1bbfebdf1a'
INVENTORY_SHA = '1e254a6659a30936376e0c80e88fe0cd159b7be0fa025f0b18cc0959fe88bd1a'
PREFIX = 'outputs/reflection_coverage_vm'
ARMS = ('control', 'reflective')
EPOCHS = (36, 42)
SIZES = {'real': 25, 'synthetic': 400, 'training_real': 73}
DOMAINS = (*SIZES, 'human_real', 'mannequin', 'glare')
ARCHIVE = ROOT / 'outputs/reflection-coverage-results.tar.gz'
EXTRACTION = ROOT / 'outputs/downloaded_reflection_coverage'
AUDIT_OUTPUT = ROOT / 'outputs/reflection_coverage_validation'
PREVIOUS = ROOT / 'outputs/downloaded_face_occlusion_continuation/outputs/face_occlusion_continuation_vm'
SETUP = ROOT / 'outputs/downloaded_face_occlusion/outputs/face_occlusion_dependencies/setup.json'
SETUP_SHA = '6af41c0662a0a91d55871c4faadb6d0810d3e366a964a4da23c5b52183ef210a'
REPLAY_SHA = 'ad6fd64aed4f773c968cac7748be0fd99a7c8679d1019e32fc64a03b13662dc7'
PREVIEW_IDS = (0, 1, 5, 6, 2, 3, 4, 17, 8, 23)


def audit_history(steps, histories, data, baseline, complete):
    require(complete.get('complete') is True and complete.get('parent_unchanged') is True,
            'Missing completion/invariance marker')
    schedule = data['core_schedule']['batches']
    supplement = data['supplemental_schedule']
    require(len(schedule) == 12 and all(len(e) == 21 for e in schedule) and len(supplement) == 252,
            'Fixed twelve-epoch schedule differs')
    require(set(steps) == set(histories) == set(complete['arms']) == set(ARMS), 'Matched arm missing')
    batches = [b for epoch in schedule for b in epoch]
    result = {'total_experiment_updates': 0, 'arms': {}}
    for arm in ARMS:
        rows, history = steps[arm], histories[arm]
        require(len(rows) == 252, 'Require exactly252 new updates per arm')
        for n, (row, batch, pair) in enumerate(zip(rows, batches, supplement), 1):
            counters = {'experiment_epoch': (n-1)//21+1, 'step': (n-1)%21+1,
                        'experiment_updates': n, 'cumulative_model_updates': 630+n,
                        'optimizer_state_step': 420+n}
            require(all(type(row.get(k)) is int and row[k] == v for k, v in counters.items()),
                    'Executed model/moment counter origins differ')
            # Independent counterpart calculation: retain the camera bit, clear the style.
            expected_pair = [pair[0] if arm == 'reflective' else pair[0]//10*10 + pair[0]%2, pair[1]]
            require(row.get('indices') == batch and row.get('supplemental_indices') == expected_pair,
                    'Executed core exposure or matched source/camera pair differs')
            require(type(row.get('supplemental_weight')) is float and row['supplemental_weight'] == .25,
                    'Supplemental weight differs')
            require(all(type(row.get(k)) in (int, float) and math.isfinite(row[k]) and row[k] >= 0
                        for k in ('supervised', 'teacher', 'supplemental', 'loss', 'pre_clip_norm')),
                    'Nonfinite/negative loss or gradient')
            total = row['supervised'] + row['teacher'] + .25 * row['supplemental']
            require(math.isclose(row['loss'], total, rel_tol=3e-6, abs_tol=1e-7), 'Total loss arithmetic differs')
        require([r.get('epoch') for r in history] == list(EPOCHS), 'Candidate check epochs differ')
        best, selected = baseline['real']['iou'], []
        for row, e in zip(history, (6, 12)):
            counters = {'epoch': 30+e, 'optimizer_updates': 630+21*e, 'additional_epoch': 20+e,
                        'fresh_optimizer_updates': 420+21*e, 'experiment_epoch': e, 'experiment_updates': 21*e}
            require(all(type(row.get(k)) is int and row[k] == v for k, v in counters.items()) and
                    row.get('arm') == arm, 'Candidate counters/arm differ')
            means = row['mean_epoch_terms']
            require(set(means) == {'loss', 'supervised', 'teacher', 'supplemental'}, 'Separate mean terms missing')
            for key in means:
                average = sum(r[key] for r in rows[(e-1)*21:e*21])/21
                require(math.isclose(means[key], average, rel_tol=0, abs_tol=1e-12), 'Mean epoch loss differs')
            require(set(DOMAINS).issubset(row), 'Candidate metric domain missing')
            real = real_gate(row['real'], baseline['real'], best)
            synthetic = retention_passes(row['synthetic'], baseline['synthetic'])
            require(all(type(row.get(k)) is bool for k in ('real_gate', 'synthetic_gate', 'selected')) and
                    (row['real_gate'], row['synthetic_gate'], row['selected']) == (real, synthetic, real and synthetic),
                    'Logged selection differs from unchanged gates')
            if real and synthetic:
                best = row['real']['iou']; selected.append(30+e)
        arm_result = complete['arms'][arm]
        require(all(type(arm_result.get(k)) is int and arm_result[k] == v for k, v in
                    {'experiment_updates': 252, 'cumulative_model_updates': 882, 'optimizer_state_step': 672}.items()) and
                arm_result.get('selected_epochs') == selected, 'Completion counters/selection differ')
        result['arms'][arm] = {'experiment_updates': 252, 'selected_epochs': selected}
        result['total_experiment_updates'] += 252
    require(type(complete.get('total_experiment_updates')) is int and complete['total_experiment_updates'] == 504,
            'Completion total differs')
    return result


def expected_files(inventory, selected):
    require(set(selected) == set(ARMS), 'Missing selected-arm registry')
    files = set(inventory) | {INVENTORY_PATH}
    files.update(f'{PREFIX}/{n}' for n in ('run.json', 'complete.json', 'baseline.json', 'source_metrics.json',
                                         'source_reflection_metrics.json', 'initial.pth', 'initial_optimizer.pth'))
    for folder, domains in [('baseline_masks', ('real', 'synthetic')), ('initial_masks', SIZES)]:
        files.update(f'{PREFIX}/{folder}/{d}/{i:04}.png' for d in domains for i in range(SIZES[d]))
    files.update(f'{PREFIX}/initial_masks/training_reflection/{i:04}.png' for i in range(280))
    for arm in ARMS:
        files.update(f'{PREFIX}/{arm}/{n}' for n in ('steps.jsonl', 'metrics.jsonl', 'restored_optimizer.json', 'final_optimizer.pth'))
        for e in EPOCHS:
            files.update((f'{PREFIX}/{arm}/epoch_{e}.pth', f'{PREFIX}/{arm}/epoch_{e}_reflection_metrics.json'))
            files.update(f'{PREFIX}/{arm}/epoch_{e}_masks/{d}/{i:04}.png' for d in SIZES for i in range(SIZES[d]))
            files.update(f'{PREFIX}/{arm}/epoch_{e}_masks/training_reflection/{i:04}.png' for i in range(280))
        if selected[arm]:
            files.add(f'{PREFIX}/{arm}/best_detector.pth')
    return files


def verify_package():
    for name, digest in PRIOR_INVENTORIES.items():
        require(sha(ROOT/name) == digest, 'Prior inventory changed')
        for member, expected in read_json(ROOT/name).items():
            canonical_member(member); require(sha(ROOT/member) == expected, 'Prior executed input changed: ' + member)
    inventory_path = ROOT/INVENTORY_PATH
    bundle = ROOT/'outputs/reflection-coverage-code.tar.gz'
    require(sha(bundle) == BUNDLE_SHA and sha(inventory_path) == INVENTORY_SHA, 'Sent reflection package changed')
    inventory = read_json(inventory_path)
    expected = {**inventory, INVENTORY_PATH: INVENTORY_SHA}; seen = set()
    with tarfile.open(bundle, 'r|gz') as tar:
        for member in tar:
            name = canonical_member(member.name)
            require(member.isfile() and name not in seen and name in expected, 'Unexpected sent member')
            require(sha_stream(tar.extractfile(member)) == expected[name], 'Sent member bytes differ'); seen.add(name)
    require(seen == set(expected), 'Sent package member missing')
    for name, digest in inventory.items():
        canonical_member(name); require(sha(ROOT/name) == digest, 'Local sent reference changed: ' + name)
    return inventory


def extract_return(archive, output, inventory):
    require(output.resolve().is_relative_to(ROOT/'outputs') and not output.exists(), 'Preserve existing extracted evidence')
    allowed = expected_files(inventory, {a: list(EPOCHS) for a in ARMS})
    directories = {p.as_posix() for n in allowed for p in Path(n).parents
                   if p.as_posix() == PREFIX or p.as_posix().startswith(PREFIX+'/')}
    provenance = {**inventory, INVENTORY_PATH: INVENTORY_SHA}
    hashes, seen, total = {}, set(), 0
    output.mkdir(parents=True)
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            name = canonical_member(member.name)
            require(name not in seen and len(seen) < 5000 and (member.isfile() or member.isdir()),
                    'Duplicate path, excessive members or unsupported archive type'); seen.add(name)
            if member.isdir():
                require(name in directories, 'Unexpected returned directory'); continue
            require(name in allowed, 'Unexpected returned file')
            if name in provenance:
                limit = (ROOT/name).stat().st_size
            elif name.endswith('optimizer.pth'):
                limit = 128*1024*1024
            elif name.endswith('.pth'):
                limit = 64*1024*1024
            elif name.endswith('.png'):
                limit = 512*1024
            else:
                limit = 2*1024*1024
            total += member.size
            require(0 <= member.size <= limit and total <= 1280*1024*1024, 'Returned size exceeds bounded inputs/results')
            destination = (output/name).resolve()
            require(destination.is_relative_to(output.resolve()), 'Extraction escapes destination')
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with tar.extractfile(member) as stream, destination.open('xb') as saved:
                for chunk in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(chunk); saved.write(chunk)
            hashes[name] = digest.hexdigest()
            if name in provenance:
                require(hashes[name] == provenance[name], 'Returned provenance differs: ' + name)
    require(set(provenance).issubset(hashes), 'Returned provenance missing')
    return hashes


def recount_supported(predictions, cases):
    require(set(predictions) == set(cases) and cases, 'Missing/extra supplemental masks')
    tp = fp = fn = visible = covered = empty = clear = clear_errors = ignored = 0
    records = []
    for i in sorted(cases):
        target = np.asarray(cases[i]['mask'])[0].astype(bool)
        valid = np.asarray(cases[i]['valid'])[0].astype(bool)
        p = predictions[i]
        require(p.dtype == np.bool_ and p.ndim == 2 and p.shape == target.shape == valid.shape and
                valid.any() and not (target & ~valid).any(), 'Invalid supported raw prediction/target')
        a = int(np.count_nonzero(p & target & valid)); b = int(np.count_nonzero(p & ~target & valid))
        c = int(np.count_nonzero(~p & target & valid)); area = int(np.count_nonzero(~target & valid))
        padding = int(np.count_nonzero(p & ~valid)); has = bool(target.any()); pred = bool((p & valid).any())
        tp += a; fp += b; fn += c; visible += area; covered += has; empty += has and not pred
        clear += not has; clear_errors += not has and pred; ignored += padding
        records.append({'case_id': i, 'tp': a, 'fp': b, 'fn': c, 'visible': area, 'ignored_positive_pixels': padding})
    return {'iou': tp/max(1, tp+fp+fn), 'missed_fraction': fn/max(1, tp+fn),
            'visible_false_positive': fp/max(1, visible), 'covered_cases': covered, 'empty_mask_cases': empty,
            'negative_cases': clear, 'negative_false_positive_cases': clear_errors,
            'ignored_positive_pixels': ignored, 'records': records,
            'scope': 'Training-only fixture diagnostics; excluded from selection'}


def checkpoint_audit(payload, initial, run, row):
    import torch
    expected = {k: v for k, v in initial.items() if k != 'model'}
    expected.update({k: row[k] for k in ('epoch', 'optimizer_updates', 'additional_epoch', 'fresh_optimizer_updates',
                                       'experiment_epoch', 'experiment_updates')})
    expected.update(reflection_metadata=run, reflection_arm=row['arm'], selection=row)
    require(set(payload) == set(expected)|{'model'} and all(payload[k] == v for k, v in expected.items()),
            'Checkpoint metadata differs')
    state, before = payload['model'], initial['model']
    require(set(state) == set(before) and not any(k.startswith(('generator.', 'segmenter.')) for k in state),
            'Detector tensor membership differs')
    require(all(isinstance(v, torch.Tensor) and v.shape == before[k].shape and v.dtype == before[k].dtype and
                torch.isfinite(v).all() for k, v in state.items()), 'Invalid checkpoint tensor')
    frozen = [k for k in state if k.startswith('reference_visible_head.') or
              k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))]
    require(frozen and all(torch.equal(state[k], before[k]) for k in frozen), 'Frozen source state changed')
    changes = {p: sum(not torch.equal(v, before[k]) for k, v in state.items() if k.startswith(p))
               for p in ('network.encoder.', 'network.decoder.', 'network.segmentation_head.')}
    require(all(changes.values()), 'Declared trainable component unchanged')
    return {'frozen_tensors_verified': len(frozen), 'changed_tensors': changes}


def audit_final_optimizer(payload, groups, model_sha, initial_optimizer):
    import torch
    validate_optimizer(initial_optimizer, groups, MODEL_SHA)
    fixed = {'model_sha256': model_sha, 'experiment_updates': 252,
             'optimizer_state_step': 672, 'cumulative_model_updates': 882}
    require(set(payload) == set(fixed)|{'state'} and all(type(payload.get(k)) is type(v) and payload[k] == v
                                                       for k, v in fixed.items()), 'Final optimizer binding/counters differ')
    state, source = payload['state'], initial_optimizer['state']
    require(set(state) == {'state', 'param_groups'} and state['param_groups'] == source['param_groups'],
            'Saved optimizer groups/settings changed')
    ids = [i for g in state['param_groups'] for i in g['params']]
    parameters = [p for g in groups for p in g['params']]
    require(set(state['state']) == set(ids) and len(ids) == len(parameters), 'Final optimizer parameter states differ')
    changed = 0
    for i, p in zip(ids, parameters):
        moment = state['state'][i]
        require(set(moment) == {'step', 'exp_avg', 'exp_avg_sq'} and isinstance(moment['step'], torch.Tensor) and
                moment['step'].numel() == 1 and torch.isfinite(moment['step']).all() and moment['step'].item() == 672,
                'Final optimizer lifetime step differs')
        require(all(isinstance(moment[k], torch.Tensor) and moment[k].shape == p.shape and moment[k].dtype == p.dtype and
                    torch.isfinite(moment[k]).all() for k in ('exp_avg', 'exp_avg_sq')) and (moment['exp_avg_sq'] >= 0).all(),
                'Invalid final optimizer moment')
        changed += int(any(not torch.equal(moment[k], source['state'][i][k]) for k in ('exp_avg', 'exp_avg_sq')))
    require(changed > 0, 'Final optimizer moments never changed')
    return {'parameter_states_verified': len(ids), 'parameter_elements': sum(p.numel() for p in parameters),
            'step_per_parameter': 672, 'changed_moment_states': changed, 'model_sha256': model_sha, 'optimizer_updates_locally': 0}


def validate_run(run, inventory):
    require(sha(SETUP) == SETUP_SHA, 'Previously verified VM setup changed')
    setup = read_json(SETUP)
    fixed = {'preflight': False, 'arms': list(ARMS), 'source_model_sha256': MODEL_SHA,
        'source_optimizer_sha256': OPTIMIZER_SHA, 'source_model_updates': 630, 'source_optimizer_step': 420,
        'optimizer_reset': False, 'epochs_per_arm': 12, 'updates_per_arm': 252, 'check_epochs': [6, 12],
        'data_manifest_sha256': DATA_SHA, 'data_audit_sha256': AUDIT_SHA, 'supplemental_pixels_sha256': PIXELS_SHA,
        'inventory_sha256': INVENTORY_SHA, 'script_sha256': inventory['scripts/train_reflection_coverage_vm.py'],
        'module_sha256': inventory['reflection_coverage.py'], 'encoder_lr': 1e-5, 'decoder_head_lr': 1e-4,
        'weight_decay': 1e-4, 'clip': 1., 'seed': 42, 'core_batch_size': 8, 'supplemental_batch_size': 2,
        'background_weight': .25, 'hard_fraction': .1, 'core_consistency_weight': 1., 'supplemental_weight': .25,
        'new_fixture_teacher_weight': 0., 'original_parent_sha256': PARENT_SHA, 'protocol_sha256': PROTOCOL_SHA,
        'replay_sha256': REPLAY_SHA, 'quarantined_sources': [67, 78, 107, 118, 121, 122, 160, 177], 'core_replacements': 49,
        'supplemental_supervision': 'Control two clear frames; reflective one reflection plus one matched clear frame. Valid-support reductions only.',
        'teacher_policy': 'Original parent on core synthetic samples only; novel fixtures have direct synthetic truth.',
        'validation_policy': 'Original real/synthetic gates at threshold0.5; all new fixture scores are training diagnostics.',
        'test_split': 'No model score; prior development inspected test images. Identity/pretraining overlap unresolved.',
        'dependencies': setup['packages'], 'setup_sha256': SETUP_SHA, 'torch': setup['torch']}
    require(set(run) == set(fixed)|{'python', 'gpu', 'opencv'}, 'Run metadata fields differ')
    for key, value in fixed.items():
        require(type(run[key]) is type(value) and run[key] == value, 'Fixed run field differs: ' + key)
    require(all(isinstance(run[k], str) and run[k] for k in ('python', 'gpu', 'opencv')), 'Missing runtime strings')


def audit():
    import torch
    from PIL import Image, ImageDraw
    require(ARCHIVE.is_file(), 'VM return archive absent: ' + str(ARCHIVE))
    require(not AUDIT_OUTPUT.exists(), 'Preserve existing audit evidence')
    transfer = ARCHIVE.with_name(ARCHIVE.name+'.sha256')
    digest = sha(ARCHIVE)
    require(transfer.is_file() and transfer.read_bytes() == (digest+'  '+ARCHIVE.name+'\n').encode(),
            'Download the matching LF checksum; transferred archive/checksum differs')
    inventory = verify_package()
    _, rows, datasets = verified_inputs()
    hashes = extract_return(ARCHIVE, EXTRACTION, inventory); returned = EXTRACTION/PREFIX
    run = read_json(returned/'run.json'); validate_run(run, inventory)
    data = read_json(ROOT/DATA_PATH/'manifest.json')
    complete = read_json(returned/'complete.json'); baseline = read_json(returned/'baseline.json')
    source = read_json(returned/'source_metrics.json')
    require(set(baseline) == set(source) == set(DOMAINS), 'Baseline/source metric domains differ')
    previous_members = read_json(ROOT/'outputs/face_occlusion_continuation_validation/members.json')
    require(sha(PREVIOUS/'baseline.json') == previous_members['outputs/face_occlusion_continuation_vm/baseline.json'] and
            baseline == read_json(PREVIOUS/'baseline.json'), 'Original common baseline changed')
    steps = {a: read_lines(returned/a/'steps.jsonl') for a in ARMS}
    histories = {a: read_lines(returned/a/'metrics.jsonl') for a in ARMS}
    log = audit_history(steps, histories, data, baseline, complete)
    require(set(hashes) == expected_files(inventory, {a: log['arms'][a]['selected_epochs'] for a in ARMS}),
            'Returned file membership differs')
    require(hashes[f'{PREFIX}/initial.pth'] == MODEL_SHA and hashes[f'{PREFIX}/initial_optimizer.pth'] == OPTIMIZER_SHA,
            'Copied source model/moments differ')
    for arm in ARMS:
        require(hashes[f'{PREFIX}/{arm}/epoch_42.pth'] == complete['arms'][arm]['final_sha256'] and
                hashes[f'{PREFIX}/{arm}/final_optimizer.pth'] == complete['arms'][arm]['final_optimizer_sha256'],
                'Final model/optimizer file binding differs')
    require(complete['promotion_requires_visual_review'] is True and math.isfinite(complete['seconds']) and
            complete['seconds'] > 0 and type(complete['peak_cuda_allocated_bytes']) is int and
            type(complete['peak_cuda_reserved_bytes']) is int and
            0 < complete['peak_cuda_allocated_bytes'] <= complete['peak_cuda_reserved_bytes'], 'Runtime/review marker differs')
    initial = torch.load(returned/'initial.pth', map_location='cpu', weights_only=True); validate_source(initial)
    same_scores(source, initial['selection'], 'source30 checkpoint')
    def trainable(prefix):
        return [v for k, v in initial['model'].items() if k.startswith(prefix) and
                not k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))]
    groups = [{'params': trainable('network.encoder.'), 'lr': 1e-5},
              {'params': trainable('network.decoder.')+trainable('network.segmentation_head.'), 'lr': 1e-4}]
    optimizer = torch.load(returned/'initial_optimizer.pth', map_location='cpu', weights_only=True)
    initial_optimizer = validate_optimizer(optimizer, groups, MODEL_SHA)
    pixels = torch.load(ROOT/DATA_PATH/'pixels.pth', map_location='cpu', weights_only=True)['pixels']
    targets = {d: [ds[i][1][0].numpy().astype(bool) for i in range(len(ds))] for d, ds in datasets.items()}
    def saved_scores(folder, domains):
        predictions = {d: [binary(folder/d/f'{i:04}.png') for i in range(len(targets[d]))] for d in domains}
        scores = {d: recount(predictions[d], targets[d]) for d in domains}
        return subsets(scores, predictions['real'], targets['real'], rows), predictions
    def saved_reflections(folder, logged):
        calculated = recount_supported({i: binary(folder/f'{i:04}.png') for i in pixels}, pixels)
        require(set(calculated) == set(logged), 'Fixture metric fields differ')
        for key in calculated:
            if type(calculated[key]) is float:
                require(type(logged[key]) in (int, float) and math.isclose(calculated[key], logged[key], rel_tol=0, abs_tol=1e-12),
                        'Supported fixture metric recount differs: ' + key)
            else:
                require(calculated[key] == logged[key], 'Supported fixture counts/records differ: ' + key)
        return calculated
    base_scores, base_predictions = saved_scores(returned/'baseline_masks', ('real', 'synthetic'))
    source_scores, source_predictions = saved_scores(returned/'initial_masks', SIZES)
    same_scores(base_scores, baseline, 'original parent'); same_scores(source_scores, source, 'source30')
    fixture_source = saved_reflections(returned/'initial_masks'/'training_reflection', read_json(returned/'source_reflection_metrics.json'))
    count = 425+498+280; candidates = {}; previews = {}; checks = {}; final_optimizers = {}
    for arm in ARMS:
        require(read_json(returned/arm/'restored_optimizer.json') == initial_optimizer, 'Logged restored moments differ')
        candidates[arm] = []
        for row in histories[arm]:
            e = row['epoch']; key = f'{arm}/{e}'; path = returned/arm/f'epoch_{e}.pth'
            payload = torch.load(path, map_location='cpu', weights_only=True)
            checks[key] = {'sha256': sha(path), **checkpoint_audit(payload, initial, run, row)}
            actual, predictions = saved_scores(returned/arm/f'epoch_{e}_masks', SIZES)
            same_scores(actual, row, key)
            fixture = saved_reflections(returned/arm/f'epoch_{e}_masks'/'training_reflection',
                                       read_json(returned/arm/f'epoch_{e}_reflection_metrics.json'))
            count += 778; previews[key] = predictions['real']
            candidates[arm].append({'epoch': e, 'selected': row['selected'], **actual, 'training_reflection': fixture})
            if e == 42:
                final = torch.load(returned/arm/'final_optimizer.pth', map_location='cpu', weights_only=True)
                final_optimizers[arm] = audit_final_optimizer(final, groups, sha(path), optimizer); del final
            del payload
        if log['arms'][arm]['selected_epochs']:
            e = log['arms'][arm]['selected_epochs'][-1]; row = next(r for r in histories[arm] if r['epoch'] == e)
            best = torch.load(returned/arm/'best_detector.pth', map_location='cpu', weights_only=True)
            checkpoint_audit(best, initial, run, row)
            require(hashes[f'{PREFIX}/{arm}/best_detector.pth'] == hashes[f'{PREFIX}/{arm}/epoch_{e}.pth'],
                    'Best checkpoint bytes differ from last eligible candidate'); del best
    require(count == 4315, 'Incomplete saved-mask recount')
    AUDIT_OUTPUT.mkdir()
    sheet = Image.new('RGB', (1024, len(PREVIEW_IDS)*148), 'white')
    for line, i in enumerate(PREVIEW_IDS):
        x, _ = datasets['real'][i]
        tiles = [Image.fromarray(np.rint(x.permute(1, 2, 0).numpy()*255).astype('uint8'))]
        masks = [targets['real'][i], base_predictions['real'][i], source_predictions['real'][i]]
        masks.extend(previews[f'{a}/{e}'][i] for e in EPOCHS for a in ARMS)
        tiles.extend(Image.fromarray(m.astype('uint8')*255).convert('RGB') for m in masks)
        ImageDraw.Draw(sheet).text((2, line*148+2),
            f'Validation {i}: input | target | parent | source30 | control36 | reflection36 | control42 | reflection42', fill='black')
        for j, tile in enumerate(tiles):sheet.paste(tile.resize((128, 128)), (j*128, line*148+20))
    sheet.save(AUDIT_OUTPUT/'preview.png')
    evidence = {'archive_sha256': digest, 'archive_bytes': ARCHIVE.stat().st_size, 'checksum_verified': True,
        'script_sha256': sha(__file__), 'bundle_sha256': BUNDLE_SHA, 'log_audit': log,
        'saved_masks_recounted': count, 'baseline': base_scores, 'source': source_scores,
        'source_training_reflection': fixture_source, 'candidates': candidates, 'checkpoint_checks': checks,
        'source_optimizer': initial_optimizer, 'final_optimizers': final_optimizers,
        'preview_validation_indices': list(PREVIEW_IDS), 'runtime': complete,
        'optimizer_updates_locally': 0, 'optimizer_constructed': False, 'promoted': False,
        'limitations': ['Checkpoint-to-mask CPU reproduction and visual review pending',
            'New fixture metrics are training-only; simplified reflections do not establish real transfer',
            'Intermediate/best optimizer snapshots not returned',
            'Remote parent invariance is an executed-code/log claim; its VM tensors were not returned',
            'External pretraining overlap and identity generalization unresolved; reused development validation']}
    (AUDIT_OUTPUT/'members.json').write_text(json.dumps(hashes, indent=2)+'\n')
    (AUDIT_OUTPUT/'results.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps({'saved_masks_recounted': count, **log, 'promoted': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-package', action='store_true')
    args = parser.parse_args()
    if args.verify_package:
        print(json.dumps({'sent_members_verified': len(verify_package())+1, 'VM_return_scored': False, 'optimizer_updates_locally': 0}))
    else:
        audit()

"""Independent audit of the matched VM loss experiment; no local optimization."""
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
    canonical_member, read_json, read_lines, require, same_scores, subsets, verified_inputs,
)
from scripts.package_face_occlusion_vm import sha, sha_stream
from scripts.train_coverage_vm import real_gate, PROTOCOL_SHA, PARENT_SHA
from detector_replay import retention_passes
from scripts.evaluate_coverage_results import binary, recount
from face_occlusion_focus import validate_source, validate_optimizer
from scripts.train_face_occlusion_focus_vm import MODEL_SHA, OPTIMIZER_SHA, OLD_INVENTORIES, MAP_PATH, PRIORITY_PATH

BUNDLE_SHA = '3c0834de0c89b5189170a9327a6bfb3112a8c11f03100229122733e73036892c'
INVENTORY_SHA = '91b23aaa05c06c377a5c50c3c0bf53a5008e7ec7bd0d5ae657684d81b0b05471'
INVENTORY_PATH = 'outputs/face_occlusion_focus_bundle_v1/inventory.json'
PREFIX = 'outputs/face_occlusion_focus_vm'
ARMS = {'control': 0., 'component_focus': .25}
EPOCHS = (35, 40)
SIZES = {'real': 25, 'synthetic': 400, 'training_real': 73}
DOMAINS = (*SIZES, 'human_real', 'mannequin', 'glare')
ARCHIVE = ROOT / 'outputs/face-occlusion-focus-results.tar.gz'
EXTRACTION = ROOT / 'outputs/downloaded_face_occlusion_focus'
AUDIT_OUTPUT = ROOT / 'outputs/face_occlusion_focus_validation'
PREVIOUS = ROOT / 'outputs/downloaded_face_occlusion_continuation/outputs/face_occlusion_continuation_vm'
SETUP = ROOT / 'outputs/downloaded_face_occlusion/outputs/face_occlusion_dependencies/setup.json'
SETUP_SHA = '6af41c0662a0a91d55871c4faadb6d0810d3e366a964a4da23c5b52183ef210a'
REPLAY_SHA = 'ad6fd64aed4f773c968cac7748be0fd99a7c8679d1019e32fc64a03b13662dc7'


def audit_history(steps, histories, schedule, baseline, complete):
    require(complete.get('complete') is True and complete.get('parent_unchanged') is True,
            'Missing completion/invariance marker')
    require(complete.get('source_model_sha256') == MODEL_SHA and
            complete.get('source_optimizer_sha256') == OPTIMIZER_SHA, 'Source model/optimizer differs')
    require(len(schedule) == 10 and all(len(e) == 21 for e in schedule), 'Fixed schedule differs')
    require(set(steps) == set(histories) == set(complete['arms']) == set(ARMS), 'Matched arm missing')
    original = [b for epoch in schedule for b in epoch]
    result = {'total_experiment_updates': 0, 'arms': {}}
    for arm, weight in ARMS.items():
        rows = steps[arm]; history = histories[arm]
        require(len(rows) == 210, 'Require exactly210 new steps per arm')
        for n, (row, batch) in enumerate(zip(rows, original), 1):
            counters = {'experiment_epoch': (n-1)//21+1, 'global_epoch': (n-1)//21+31,
                        'step': (n-1)%21+1, 'experiment_updates': n,
                        'cumulative_model_updates': 630+n, 'optimizer_state_step': 420+n}
            require(all(type(row.get(k)) is int and row[k] == v for k, v in counters.items()),
                    'Executed counter origins differ')
            require(row.get('indices') == batch, 'Executed schedule differs')
            require(type(row.get('focus_weight')) is float and row['focus_weight'] == weight, 'Auxiliary weight differs')
            require(all(type(row.get(k)) in (int,float) and math.isfinite(row[k]) and row[k] >= 0
                        for k in ('supervised','teacher','focus','loss','pre_clip_norm')), 'Nonfinite/negative loss or gradient')
            total = row['supervised'] + row['teacher'] + weight * row['focus']
            # Three scalar float32 operations on CUDA can round relative to this
            # float64 independent reconstruction; never tolerate a changed term.
            require(math.isclose(row['loss'], total, rel_tol=3e-6, abs_tol=1e-7), 'Total loss arithmetic differs')
        require([r.get('epoch') for r in history] == list(EPOCHS), 'Candidate check epochs differ')
        best = baseline['real']['iou']; selected = []
        for row, e in zip(history, (5,10)):
            counters = {'epoch': 30+e, 'optimizer_updates': 630+21*e, 'additional_epoch': 20+e,
                        'fresh_optimizer_updates': 420+21*e, 'experiment_epoch': e, 'experiment_updates': 21*e}
            require(all(type(row.get(k)) is int and row[k] == v for k,v in counters.items()) and
                    row.get('arm') == arm, 'Candidate counter/arm differs')
            means = row['mean_epoch_terms']
            require(set(means) == {'loss','supervised','teacher','focus'}, 'Separate epoch terms missing')
            for key in means:
                average = sum(r[key] for r in rows[(e-1)*21:e*21]) / 21
                require(math.isclose(means[key], average, rel_tol=0, abs_tol=1e-12), 'Mean epoch loss term differs')
            require(set(DOMAINS).issubset(row), 'Candidate metric domain missing')
            real = real_gate(row['real'], baseline['real'], best)
            synthetic = retention_passes(row['synthetic'], baseline['synthetic'])
            require(all(type(row.get(k)) is bool for k in ('real_gate','synthetic_gate','selected')) and
                    (row['real_gate'],row['synthetic_gate'],row['selected']) == (real,synthetic,real and synthetic),
                    'Logged selection differs from unchanged gates')
            if real and synthetic: best = row['real']['iou']; selected.append(30+e)
        arm_result = complete['arms'][arm]
        require(all(type(arm_result.get(k)) is int and arm_result[k] == v for k,v in
                    {'experiment_updates':210,'cumulative_model_updates':840,'optimizer_state_step':630}.items()) and
                arm_result.get('selected_epochs') == selected, 'Completion counters/selection differ')
        result['arms'][arm] = {'experiment_updates':210,'selected_epochs':selected}
        result['total_experiment_updates'] += 210
    require(type(complete.get('total_experiment_updates')) is int and complete['total_experiment_updates'] == 420,
            'Completion total updates differ')
    return result


def expected_files(inventory, *, selected):
    files = set(inventory) | {INVENTORY_PATH}
    files.update(f'{PREFIX}/{n}' for n in ('run.json','complete.json','baseline.json','source_metrics.json',
                                         'initial.pth','initial_optimizer.pth'))
    for folder, domains in [('baseline_masks', ('real','synthetic')), ('initial_masks', SIZES)]:
        files.update(f'{PREFIX}/{folder}/{d}/{i:04}.png' for d in domains for i in range(SIZES[d]))
    for arm in ARMS:
        files.update(f'{PREFIX}/{arm}/{n}' for n in ('steps.jsonl','metrics.jsonl','restored_optimizer.json','final_optimizer.pth'))
        for e in EPOCHS:
            files.add(f'{PREFIX}/{arm}/epoch_{e}.pth')
            files.update(f'{PREFIX}/{arm}/epoch_{e}_masks/{d}/{i:04}.png' for d in SIZES for i in range(SIZES[d]))
        if selected[arm]: files.add(f'{PREFIX}/{arm}/best_detector.pth')
    return files


def member_limit(name):
    canonical_member(name)
    if name == f'{PREFIX}/initial_optimizer.pth' or name in {f'{PREFIX}/{a}/final_optimizer.pth' for a in ARMS}:
        return 128*1024*1024
    if name.endswith('.pth'): return 64*1024*1024
    if name.endswith('.png'): return 512*1024
    return 2*1024*1024


def verify_package():
    for name, digest in OLD_INVENTORIES.items():
        require(sha(ROOT/name) == digest, 'Previous inventory changed')
        for relative, expected in read_json(ROOT/name).items():
            canonical_member(relative)
            require(sha(ROOT/relative) == expected, f'Previous input changed: {relative}')
    inventory_path = ROOT/INVENTORY_PATH; bundle = ROOT/'outputs/face-occlusion-focus-code.tar.gz'
    require(sha(bundle) == BUNDLE_SHA and sha(inventory_path) == INVENTORY_SHA, 'Sent focus package changed')
    inventory = read_json(inventory_path); expected = {**inventory, INVENTORY_PATH:INVENTORY_SHA}; seen = set()
    with tarfile.open(bundle, 'r|gz') as tar:
        for member in tar:
            name = canonical_member(member.name)
            require(member.isfile() and name not in seen and name in expected, 'Unexpected sent package member')
            require(sha_stream(tar.extractfile(member)) == expected[name], 'Sent member bytes differ'); seen.add(name)
    require(seen == set(expected), 'Sent package member missing')
    for name,digest in inventory.items():
        canonical_member(name); require(sha(ROOT/name) == digest, f'Local sent reference changed: {name}')
    return inventory


def extract_return(archive, output, inventory):
    require(output.resolve().is_relative_to(ROOT) and not output.exists(), 'Preserve existing extracted evidence')
    allowed = expected_files(inventory, selected={a:[35,40] for a in ARMS})
    directories = {p.as_posix() for n in allowed for p in Path(n).parents
                   if p.as_posix() == PREFIX or p.as_posix().startswith(PREFIX+'/')}
    provenance = {**inventory, INVENTORY_PATH:INVENTORY_SHA}; hashes = {}; seen = set(); total = 0
    output.mkdir(parents=True)
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            name = canonical_member(member.name)
            require(name not in seen and len(seen) < 3200 and (member.isfile() or member.isdir()),
                    'Duplicate path, excessive members or unsupported archive type'); seen.add(name)
            if member.isdir(): require(name in directories, 'Unexpected returned directory'); continue
            require(name in allowed, 'Unexpected returned file')
            total += member.size
            require(0 <= member.size <= member_limit(name) and total <= 900*1024*1024, 'Returned size exceeds fixed bound')
            destination = (output/name).resolve(); require(destination.is_relative_to(output.resolve()), 'Extraction escapes destination')
            destination.parent.mkdir(parents=True,exist_ok=True); digest = hashlib.sha256()
            with tar.extractfile(member) as stream, destination.open('xb') as saved:
                for chunk in iter(lambda: stream.read(1024*1024),b''): digest.update(chunk); saved.write(chunk)
            hashes[name] = digest.hexdigest()
            if name in provenance: require(hashes[name] == provenance[name], f'Returned provenance differs: {name}')
    require(set(provenance).issubset(hashes), 'Returned provenance missing')
    return hashes


def checkpoint_audit(payload, initial, run, row):
    import torch
    expected = {k:v for k,v in initial.items() if k != 'model'}
    expected.update({k:row[k] for k in ('epoch','optimizer_updates','additional_epoch','fresh_optimizer_updates',
                                      'experiment_epoch','experiment_updates')})
    expected.update(focus_metadata=run, focus_arm=row['arm'], selection=row)
    require(set(payload) == set(expected)|{'model'} and all(payload[k] == v for k,v in expected.items()),
            'Checkpoint metadata differs')
    state = payload['model']; before = initial['model']
    require(set(state) == set(before) and not any(k.startswith(('generator.','segmenter.')) for k in state),
            'Detector tensor membership differs')
    require(all(isinstance(v,torch.Tensor) and v.shape == before[k].shape and v.dtype == before[k].dtype
                and torch.isfinite(v).all() for k,v in state.items()), 'Invalid checkpoint tensor')
    frozen = [k for k in state if k.startswith('reference_visible_head.') or
              k.endswith(('running_mean','running_var','num_batches_tracked'))]
    require(frozen and all(torch.equal(state[k],before[k]) for k in frozen), 'Frozen source state changed')
    changes = {p:sum(not torch.equal(v,before[k]) for k,v in state.items() if k.startswith(p))
               for p in ('network.encoder.','network.decoder.','network.segmentation_head.')}
    require(all(changes.values()), 'Declared trainable component unchanged')
    return {'frozen_tensors_verified':len(frozen),'changed_tensors':changes}


def audit_final_optimizer(payload, groups, model_sha, initial_optimizer):
    import torch
    validate_optimizer(initial_optimizer,groups,MODEL_SHA)
    fixed = {'model_sha256':model_sha,'experiment_updates':210,'optimizer_state_step':630,'cumulative_model_updates':840}
    require(set(payload) == set(fixed)|{'state'} and all(type(payload.get(k)) is type(v) and payload[k] == v
                                                       for k,v in fixed.items()), 'Final optimizer binding/counters differ')
    state = payload['state']; source = initial_optimizer['state']
    require(set(state) == {'state','param_groups'} and state['param_groups'] == source['param_groups'],
            'Saved optimizer groups/settings changed')
    ids = [i for g in state['param_groups'] for i in g['params']]; parameters = [p for g in groups for p in g['params']]
    require(set(state['state']) == set(ids) and len(ids) == len(parameters), 'Final optimizer parameter states differ')
    changed = 0
    for i,p in zip(ids,parameters):
        moment = state['state'][i]
        require(set(moment) == {'step','exp_avg','exp_avg_sq'} and isinstance(moment['step'],torch.Tensor) and
                moment['step'].numel() == 1 and torch.isfinite(moment['step']).all() and moment['step'].item() == 630,
                'Final optimizer lifetime step differs')
        require(all(isinstance(moment[k],torch.Tensor) and moment[k].shape == p.shape and moment[k].dtype == p.dtype
                    and torch.isfinite(moment[k]).all() for k in ('exp_avg','exp_avg_sq')) and
                (moment['exp_avg_sq'] >= 0).all(), 'Invalid final optimizer moment')
        changed += int(any(not torch.equal(moment[k],source['state'][i][k]) for k in ('exp_avg','exp_avg_sq')))
    require(changed > 0, 'Final optimizer moments never changed')
    return {'parameter_states_verified':len(ids),'parameter_elements':sum(p.numel() for p in parameters),
            'step_per_parameter':630,'changed_moment_states':changed,'model_sha256':model_sha,'optimizer_updates_locally':0}


def validate_run(run, inventory):
    require(sha(SETUP) == SETUP_SHA, 'Previously verified VM setup changed'); setup = read_json(SETUP)
    old_face = read_json(ROOT/'outputs/face_occlusion_bundle_v1/inventory.json')
    fixed = {'preflight':False,'source_model_sha256':MODEL_SHA,'source_optimizer_sha256':OPTIMIZER_SHA,
             'source_epoch':30,'source_model_updates':630,'source_optimizer_step':420,'optimizer_reset':False,
             'optimizer_state_restored':True,'epochs_per_arm':10,'updates_per_arm':210,'arms':ARMS,'check_epochs':[5,10],
             'protocol_sha256':PROTOCOL_SHA,'replay_sha256':REPLAY_SHA,'parent_sha256':PARENT_SHA,
             'prior_inventories':OLD_INVENTORIES,'inventory_sha256':INVENTORY_SHA,
             'map_manifest_sha256':inventory[MAP_PATH],'priority_manifest_sha256':inventory[PRIORITY_PATH],
             'script_sha256':inventory['scripts/train_face_occlusion_focus_vm.py'],
             'specification_sha256':inventory['FACE_OCCLUSION_FOCUS.md'],
             'focus_module_sha256':inventory['face_occlusion_focus.py'],'adapter_sha256':old_face['face_occlusion_adapter.py'],
             'dependencies':setup['packages'],'setup_sha256':SETUP_SHA,'torch':setup['torch'],
             'encoder_lr':1e-5,'decoder_head_lr':1e-4,'weight_decay':1e-4,'clip':1.,'seed':42,
             'background_weight':.25,'consistency_weight':1.,'component_ring_radius':3,'clear_hard_fraction':.1,
             'validation_batch_size':1,'batchnorm_policy':'fixed_running_statistics_trainable_affine',
             'test_split':'Not scored; previous development work inspected it',
             'overlap_limitation':'External FaceExtraction FFHQ pretraining overlap unresolved'}
    require(set(run) == set(fixed)|{'python','gpu','opencv'}, 'Run metadata fields differ')
    for key,value in fixed.items():
        require(type(run[key]) is type(value) and run[key] == value, f'Fixed run field differs: {key}')
    require(all(isinstance(run[k],str) and run[k] for k in ('python','gpu','opencv')), 'Missing runtime strings')


def audit():
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    require(ARCHIVE.is_file(), f'VM return archive absent: {ARCHIVE}')
    require(not AUDIT_OUTPUT.exists(), 'Preserve existing audit evidence')
    inventory = verify_package(); protocol,rows,data = verified_inputs()
    hashes = extract_return(ARCHIVE,EXTRACTION,inventory); returned = EXTRACTION/PREFIX
    run = read_json(returned/'run.json'); validate_run(run,inventory)
    complete = read_json(returned/'complete.json'); baseline = read_json(returned/'baseline.json')
    source = read_json(returned/'source_metrics.json')
    require(set(baseline) == set(source) == set(DOMAINS), 'Baseline/source metric domains differ')
    previous_members = read_json(ROOT/'outputs/face_occlusion_continuation_validation/members.json')
    require(sha(PREVIOUS/'baseline.json') == previous_members['outputs/face_occlusion_continuation_vm/baseline.json'] and
            baseline == read_json(PREVIOUS/'baseline.json'), 'Original common baseline changed')
    steps = {a:read_lines(returned/a/'steps.jsonl') for a in ARMS}
    histories = {a:read_lines(returned/a/'metrics.jsonl') for a in ARMS}
    log = audit_history(steps,histories,protocol['schedules']['extended']['batches'],baseline,complete)
    require(set(hashes) == expected_files(inventory,selected={a:log['arms'][a]['selected_epochs'] for a in ARMS}),
            'Returned file membership differs')
    require(hashes[f'{PREFIX}/initial.pth'] == MODEL_SHA and hashes[f'{PREFIX}/initial_optimizer.pth'] == OPTIMIZER_SHA,
            'Copied source model/optimizer bytes differ')
    for arm in ARMS:
        require(hashes[f'{PREFIX}/{arm}/epoch_40.pth'] == complete['arms'][arm]['final_sha256'] and
                hashes[f'{PREFIX}/{arm}/final_optimizer.pth'] == complete['arms'][arm]['final_optimizer_sha256'],
                'Final model/optimizer file hash differs')
    require(complete['promotion_requires_visual_review'] is True and math.isfinite(complete['seconds']) and
            complete['seconds'] > 0 and type(complete['peak_cuda_allocated_bytes']) is int and
            type(complete['peak_cuda_reserved_bytes']) is int and
            0 < complete['peak_cuda_allocated_bytes'] <= complete['peak_cuda_reserved_bytes'], 'Runtime/review marker differs')
    initial = torch.load(returned/'initial.pth',map_location='cpu',weights_only=True); validate_source(initial)
    same_scores(source,initial['selection'],'source30 checkpoint')
    def trainable(prefix):
        return [v for k,v in initial['model'].items() if k.startswith(prefix) and
                not k.endswith(('running_mean','running_var','num_batches_tracked'))]
    groups = [{'params':trainable('network.encoder.'),'lr':1e-5},
              {'params':trainable('network.decoder.')+trainable('network.segmentation_head.'),'lr':1e-4}]
    optimizer = torch.load(returned/'initial_optimizer.pth',map_location='cpu',weights_only=True)
    initial_optimizer = validate_optimizer(optimizer,groups,MODEL_SHA)
    targets = {d:[data[d][i][1][0].numpy().astype(bool) for i in range(SIZES[d])] for d in SIZES}
    def scores(folder,domains):
        predictions = {d:[binary(folder/d/f'{i:04}.png') for i in range(SIZES[d])] for d in domains}
        result = {d:recount(predictions[d],targets[d]) for d in domains}
        subsets(result,predictions['real'],targets['real'],rows)
        return result,predictions
    base_scores,base_predictions = scores(returned/'baseline_masks',('real','synthetic'))
    source_scores,source_predictions = scores(returned/'initial_masks',SIZES)
    same_scores(base_scores,baseline,'original parent'); same_scores(source_scores,source,'source30')
    count = 425+498; candidates = {}; previews = {}; checks = {}; final_optimizers = {}
    for arm in ARMS:
        require(read_json(returned/arm/'restored_optimizer.json') == initial_optimizer, 'Logged optimizer restoration differs')
        candidates[arm] = []
        for row in histories[arm]:
            e = row['epoch']; key = f'{arm}/{e}'; path = returned/arm/f'epoch_{e}.pth'
            payload = torch.load(path,map_location='cpu',weights_only=True)
            checks[key] = {'sha256':sha(path),**checkpoint_audit(payload,initial,run,row)}
            actual,predictions = scores(returned/arm/f'epoch_{e}_masks',SIZES)
            same_scores(actual,row,key); count += 498; previews[key] = predictions['real']
            candidates[arm].append({'epoch':e,'selected':row['selected'],**actual})
            if e == 40:
                final_optimizer = torch.load(returned/arm/'final_optimizer.pth',map_location='cpu',weights_only=True)
                final_optimizers[arm] = audit_final_optimizer(final_optimizer,groups,sha(path),optimizer); del final_optimizer
            del payload
        selected = log['arms'][arm]['selected_epochs']
        if selected:
            e = selected[-1]; row = next(r for r in histories[arm] if r['epoch'] == e)
            best = torch.load(returned/arm/'best_detector.pth',map_location='cpu',weights_only=True)
            checkpoint_audit(best,initial,run,row)
            candidate = torch.load(returned/arm/f'epoch_{e}.pth',map_location='cpu',weights_only=True)
            require(all(torch.equal(v,candidate['model'][k]) for k,v in best['model'].items()), 'Best state differs from eligible candidate')
            del best,candidate
    require(count == 2915, 'Incomplete saved-mask recount')
    AUDIT_OUTPUT.mkdir(); preview_ids = [0,1,2,3,4,5,6,7,22,23]
    sheet = Image.new('RGB',(1024,len(preview_ids)*148),'white')
    for line,i in enumerate(preview_ids):
        x,_ = data['real'][i]
        tiles = [Image.fromarray(np.rint(x.permute(1,2,0).numpy()*255).astype('uint8'))]
        masks = [targets['real'][i],base_predictions['real'][i],source_predictions['real'][i]]
        masks.extend(previews[f'{a}/{e}'][i] for e in EPOCHS for a in ARMS)
        tiles.extend(Image.fromarray(m.astype('uint8')*255).convert('RGB') for m in masks)
        ImageDraw.Draw(sheet).text((2,line*148+2),
            f'Validation {i}: input | target | parent | source30 | control35 | focus35 | control40 | focus40',fill='black')
        for j,tile in enumerate(tiles): sheet.paste(tile.resize((128,128)),(j*128,line*148+20))
    sheet.save(AUDIT_OUTPUT/'preview.png')
    evidence = {'archive_sha256':sha(ARCHIVE),'archive_bytes':ARCHIVE.stat().st_size,'script_sha256':sha(__file__),
                'bundle_sha256':BUNDLE_SHA,'log_audit':log,'saved_masks_recounted':count,'baseline':base_scores,
                'source':source_scores,'candidates':candidates,'checkpoint_checks':checks,
                'source_optimizer':initial_optimizer,'final_optimizers':final_optimizers,
                'preview_validation_indices':preview_ids,'runtime':complete,'optimizer_updates_locally':0,'promoted':False,
                'limitations':['Checkpoint-to-mask CPU reproduction pending',
                    'Intermediate/best optimizer snapshots not returned',
                    'Remote original-parent invariance is an executed-code/log claim; its VM tensors were not returned',
                    'External pretraining overlap and generalization unresolved; repeatedly reused development cases']}
    (AUDIT_OUTPUT/'members.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (AUDIT_OUTPUT/'results.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'saved_masks_recounted':count,**log,'promoted':False}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-package',action='store_true')
    args = parser.parse_args()
    if args.verify_package: print(json.dumps({'members':len(verify_package())+1,'optimizer_updates_locally':0}))
    else: audit()

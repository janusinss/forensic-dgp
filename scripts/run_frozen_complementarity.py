"""Fixed CPU-only frozen-head feasibility; no optimizer, router or promotion."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from frozen_complementarity import counts, aggregate, fixed_compositions, diagnostic_oracles, conflicting_inputs, frozen_predictions, require
from supported_real_data import load_supported_manifest, read_item, sha

REGISTRY = ROOT/'dataset/detector_supported_review_v1/manifest.json'
REPLAY = ROOT/'outputs/downloaded_coverage/outputs/coverage_training_vm/replay_pixels.pth'
FIXTURES = ROOT/'outputs/reflection_coverage_data_v1'
RETURNED = ROOT/'outputs/downloaded_reflection_coverage'
PREFIX = 'outputs/reflection_coverage_vm'
PARENT = ROOT/'outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth'
CANDIDATE = RETURNED/PREFIX/'reflective/epoch_42.pth'
PROTOCOL = ROOT/'outputs/frozen_complementarity_protocol_v1/protocol.json'
OUTPUT = ROOT/'outputs/frozen_complementarity_v1'
MODELS = {'parent': PARENT, 'candidate': CANDIDATE}
EXPECTED = {'registry': '860ea1dfba8fa442cab19bf3ab41f6a678109dcdb067c38ec0bd79ce43b51ace',
            'data_audit': '308e5ca5b00bf8c52b36eba78605e41d188b754044307fd54d4f39edf16cbfe9',
            'replay': 'ad6fd64aed4f773c968cac7748be0fd99a7c8679d1019e32fc64a03b13662dc7',
            'fixtures': '6696cee3a3acf48049f2f121a2a6328625c4351f558deabb5475da7fe948baf9',
            'pixels': 'ab47a88d79fe8d300ca92d572a2c1a05b92fa273b9577cb209df99e692009cc1',
            'parent': 'c2d4cd180226413af63bcfc2a3e17461cfa95f87aff9a8998b893fc3afc42e93',
            'candidate': 'a51f20e8fa12cee7dec574195debc5ada9b8cfeeb289b12bad11a6d39a9acf25'}
METHODS = ('parent', 'candidate', 'union', 'intersection', 'pool_oracle', 'dominance_oracle', 'pixel_oracle')
CODE = ('frozen_complementarity.py', 'scripts/run_frozen_complementarity.py',
        'tests/test_frozen_complementarity.py', 'tests/test_frozen_complementarity_protocol.py',
        'FROZEN_COMPLEMENTARITY_PROTOCOL.md', 'supported_real_data.py',
        'face_occlusion_adapter.py', 'completion_inference.py', 'completion.py',
        'scripts/train_face_occlusion_vm.py')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def current_core_ids(core, cache_ids, real_count=73):
    require(isinstance(core, dict) and core.get('batches') and type(real_count) is int and real_count > 0,
            'Existing core schedule required')
    quarantine = set(core['quarantined_sources']); used = set()
    require(all(type(i) is int and i >= 0 for i in quarantine), 'Invalid source quarantine')
    for epoch in core['batches']:
        for batch in epoch:
            require(len(batch) == 8 and all(type(i) is int and i >= 0 for i in batch), 'Invalid fixed core batch')
            for index in batch:
                if index >= real_count:
                    case = index-real_count
                    require(case in cache_ids and case//10 not in quarantine, 'Quarantined or unregistered replay input')
                    used.add(case)
    return sorted(used)


def binary(path):
    with Image.open(path) as image:
        array = np.array(image)
    require(array.shape == (256, 256) and array.dtype == np.uint8 and np.isin(array, [0, 255]).all(), 'Frozen mask artifact differs')
    return array.astype(bool)


def prepare_inputs():
    audit_path = ROOT/'outputs/supported_real_dataset_validation_v1/verification.json'
    checks = {'registry': REGISTRY, 'data_audit': audit_path, 'replay': REPLAY,
              'fixtures': FIXTURES/'manifest.json', 'pixels': FIXTURES/'pixels.pth', **MODELS}
    require(all(sha(p) == EXPECTED[k] for k, p in checks.items()), 'Frozen input or checkpoint binding changed')
    verification = read_json(audit_path); metadata, all_rows = load_supported_manifest(REGISTRY)
    require(verification['complete'] is True and verification['manifest_sha256'] == EXPECTED['registry']
            and verification['original_training_tensors_equal'] == 73
            and all(sha(ROOT/p) == h for p, h in metadata['input_sha256'].items()), 'Independent data or annotation lineage changed')
    rows = [r for r in all_rows if r['split'] == 'train']
    require(len(rows) == 83 and Counter(r['kind'] for r in rows) == {'covered': 51, 'uncovered': 32}, 'Training-only real membership differs')
    data = read_json(FIXTURES/'manifest.json')
    old_protocol_path = ROOT/'outputs/coverage_protocol_v1/protocol.json'; old_protocol = read_json(old_protocol_path)
    require(sha(old_protocol_path) == data['protocol_sha256']
            and data['replay_sha256'] == EXPECTED['replay']
            and all(sha(ROOT/p) == h for p, h in {**old_protocol['input_hashes'], **data['input_hashes']}.items()),
            'Existing training protocol or fixture provenance differs')
    split_path = ROOT/'outputs/downloaded_phase4/outputs/phase4_with_progress/split.json'
    train_paths = set(read_json(split_path)['train'])
    require(len(old_protocol['sources']) == 200 and all(s['path'] in train_paths and sha(ROOT/s['path']) == s['sha256']
            for s in old_protocol['sources']), 'Replay source left original training partition or changed')
    require(all(s['path'] in train_paths and sha(ROOT/s['path']) == s['sha256'] for s in data['sources']), 'Fixture source left training partition')
    replay = torch.load(REPLAY, map_location='cpu', weights_only=True)
    core_ids = current_core_ids(data['core_schedule'], set(replay))
    require(len(replay) == 638 and len(core_ids) == 610
            and Counter('covered' if i%5 else 'uncovered' for i in core_ids) == {'covered': 360, 'uncovered': 250}
            and data['core_schedule']['quarantined_sources'] == [67, 78, 107, 118, 121, 122, 160, 177], 'Existing safe replay cohort differs')
    fixture_cache = torch.load(FIXTURES/'pixels.pth', map_location='cpu', weights_only=True)
    require(fixture_cache['format'] == 'dgp-reflection-coverage-pixels-v1'
            and set(fixture_cache['pixels']) == set(range(280)), 'Existing fixture membership differs')
    repro_path = ROOT/'outputs/reflection_coverage_validation/reproduction.json'
    audit_return_path = ROOT/'outputs/reflection_coverage_validation/results.json'
    members_path = ROOT/'outputs/reflection_coverage_validation/members.json'
    repro = read_json(repro_path); return_audit = read_json(audit_return_path); members = read_json(members_path)
    require(repro['complete'] is True and return_audit['checksum_verified'] is True
            and repro['audit_sha256'] == sha(audit_return_path)
            and repro['reproductions']['reflective/42']['mask_reproduction']['training_real']['all_exact'] is True
            and repro['fixture_reproductions']['reflective/42']['mask_reproduction']['all_exact'] is True
            and return_audit['checkpoint_checks']['reflective/42']['sha256'] == EXPECTED['candidate'], 'Reused predictions lack complete prior reproduction')
    old_v2_path = ROOT/'dataset/detector_expanded_review_v2/manifest.json'
    glare_path = ROOT/'dataset/detector_glare_review_v3/manifest.json'
    require(sha(old_v2_path) == read_json(glare_path)['parent_manifest_sha256'], 'Existing lens-only label lineage differs')
    old_v2 = {r['image_sha256']: r for r in read_json(old_v2_path)['records']}
    items = []; serialized = []; lenses = {}; cached = {}; raw_bindings = {}
    def add(info, rgb, target, valid):
        require(rgb.shape == (256,256,3) and rgb.dtype == np.uint8, 'Fixed byte RGB input required')
        counts(target, target, valid)
        items.append({**info, 'rgb': rgb, 'target': target, 'valid': valid})
        serialized.append({**info, 'input_byte_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                           'target_byte_sha256': hashlib.sha256(target.tobytes()).hexdigest(),
                           'valid_byte_sha256': hashlib.sha256(valid.tobytes()).hexdigest(),
                           'target_pixels': int(target.sum()), 'valid_pixels': int(valid.sum())})
    for i, row in enumerate(rows):
        rgb, target, valid, _ = read_item(row, 256); name = f'real/{i:04}'
        add({'id': name, 'pool': 'real', 'real_index': i, 'source_sha256': row['source_sha256'],
             'image_sha256': row['image_sha256'], 'stratum': row['data_origin']+'/'+row['kind'],
             'source_pool': 'reviewed_real', 'new_source_index': row.get('source_index')}, rgb, target, valid)
        if row.get('glare_stratum') == 'strong_lens_reflection':
            old = old_v2[row['image_sha256']]; file = old_v2_path.parent/old['mask']
            require(sha(file) == old['mask_sha256'], 'Pre-glare target changed')
            lenses[name] = target & ~binary(file); raw_bindings[file.relative_to(ROOT).as_posix()] = sha(file)
        elif row.get('source_index') in (171,216,348,374):
            lenses[name] = target.copy()
        if i < 73:
            relative = f'{PREFIX}/reflective/epoch_42_masks/training_real/{i:04}.png'
            file = RETURNED/relative
            require(sha(file) == members[relative], 'Reused real training mask changed')
            cached[name] = binary(file); raw_bindings[file.relative_to(ROOT).as_posix()] = sha(file)
    for index in core_ids:
        image, mask = replay[index]
        require(image.dtype == torch.uint8 and image.shape == (3,256,256) and mask.shape == (1,256,256)
                and bool(torch.isin(mask, torch.tensor([0,1], dtype=mask.dtype)).all()), 'Cached core input/target differs')
        source = old_protocol['sources'][index//10]
        add({'id': f'replay/{index:04}', 'pool': 'replay', 'replay_index': index,
             'source_sha256': source['sha256'], 'source_pool': source['source'],
             'stratum': ('none','lower','eyes','object','irregular')[index%5]+'/'+str(index%10>=5)},
            image.permute(1,2,0).numpy(), mask[0].numpy().astype(bool), np.ones((256,256),bool))
    for row in data['cases']:
        index = row['case_id']; case = fixture_cache['pixels'][index]
        require(all(hashlib.sha256(t.numpy().tobytes()).hexdigest() == row['pixel_sha256'][k] for k,t in case.items()), 'Reviewed fixture pixels differ')
        add({'id': f'reflection/{index:04}', 'pool': 'reflection', 'fixture_index': index,
             'source_sha256': row['source_sha256'], 'source_pool': data['sources'][index//10]['source_pool'],
             'stratum': row['style']+'/'+str(row['degraded'])}, case['input'].permute(1,2,0).numpy(),
            case['mask'][0].numpy().astype(bool), case['valid'][0].numpy().astype(bool))
        relative = f'{PREFIX}/reflective/epoch_42_masks/training_reflection/{index:04}.png'; file = RETURNED/relative
        require(sha(file) == members[relative], 'Reused reflection fixture mask changed')
        cached[f'reflection/{index:04}'] = binary(file); raw_bindings[file.relative_to(ROOT).as_posix()] = sha(file)
    require(len(items) == 973 and len(cached) == 353 and len(lenses) == 6
            and sum(int(v.sum()) for v in lenses.values()) == 21853, 'Fixed case or lens-only membership differs')
    preview = [next(r['id'] for r in items if r.get('new_source_index') == i) for i in (171,216,348,374)]
    preview += [f'real/{i:04}' for i,r in enumerate(rows) if r.get('glare_stratum') == 'strong_lens_reflection']
    preview += [next(r['id'] for r in items if r.get('new_source_index') == i) for i in (207,208)]
    preview += [f'replay/{next(i for i in core_ids if i%10==k):04}' for k in (9,5)]
    input_paths = (*checks.values(), old_protocol_path, split_path, repro_path, audit_return_path, members_path,
                   old_v2_path, glare_path, ROOT/'OCCLUSION_POLICY_V3.md')
    expected = {'format': 'dgp-frozen-complementarity-protocol-v1', 'date': '2026-10-02',
                'input_sha256': {p.relative_to(ROOT).as_posix():sha(p) for p in input_paths},
                'reused_artifact_sha256': raw_bindings, 'code_sha256': {p:sha(ROOT/p) for p in CODE},
                'cases': serialized, 'counts': {'real':83,'replay':610,'reflection':280},
                'reused_predictions':353, 'new_forward_images':1593, 'device':'cpu', 'batch_size':8, 'threshold':.5,
                'model_sha256': {k:EXPECTED[k] for k in MODELS}, 'methods':list(METHODS), 'preview_ids':preview,
                'quarantined_sources':data['core_schedule']['quarantined_sources'], 'lens_target_pixels':21853,
                'oracles_deployable':False, 'held_out_forward_images':0, 'optimizer_updates_locally':0, 'promoted':False}
    return expected, items, lenses, cached


def systems_for(item, predictions):
    p, c = (predictions[k][item['id']] for k in ('parent','candidate'))
    oracle = diagnostic_oracles(p,c,item['target'],item['valid'],item['pool'])
    return {'parent':p, 'candidate':c, **fixed_compositions(p,c),
            **{k:oracle[k] for k in ('pool_oracle','dominance_oracle','pixel_oracle')}}, oracle['candidate_dominates']


def scores_for(items, predictions):
    grouped = {m:defaultdict(list) for m in METHODS}; records = []; switches = Counter(); lenses = {}
    for item in items:
        systems, dominates = systems_for(item,predictions)
        switches[item['pool']+'/'+('candidate' if dominates else 'parent')] += 1
        counted = {m:counts(p,item['target'],item['valid']) for m,p in systems.items()}
        records.append({'id':item['id'],'candidate_dominates':dominates,'counts':counted})
        groups = (item['pool'], item['pool']+'/stratum/'+item['stratum'], item['pool']+'/source/'+item['source_pool'])
        for method in METHODS:
            for group in groups: grouped[method][group].append(counted[method])
    return {m:{g:aggregate(v) for g,v in groups.items()} for m,groups in grouped.items()}, records, dict(switches)


def overlay(rgb, prediction, target, valid):
    a = rgb.astype('float32').copy()
    for where,color in (((prediction&target&valid),(45,190,90)),
                        ((prediction&~target&valid),(230,65,65)),((~prediction&target&valid),(235,195,40))):
        a[where] = .45*a[where]+.55*np.array(color)
    a[~valid] = .65*a[~valid]+.35*np.array([100,80,135])
    return Image.fromarray(np.clip(a,0,255).astype('uint8'))


def preview(items, predictions, ids):
    columns = ('input','target','parent','candidate','union','dominance_oracle'); by_id={r['id']:r for r in items}
    sheet = Image.new('RGB',(6*192,36+10*222),(244,242,236)); draw=ImageDraw.Draw(sheet)
    for j,name in enumerate(columns): draw.text((j*192+4,4),name+(' TARGET USED' if 'oracle' in name else ''),fill=(30,40,50))
    draw.text((4,20),'Green=TP red=FP yellow=FN purple=ignored. Training diagnostic only.',fill=(30,40,50))
    for i,name in enumerate(ids):
        item=by_id[name]; systems,_=systems_for(item,predictions); y=36+i*222
        for j,column in enumerate(columns):
            tile = Image.fromarray(item['rgb']) if column=='input' else overlay(item['rgb'],item['target'] if column=='target' else systems[column],item['target'],item['valid'])
            sheet.paste(tile.resize((192,192),Image.Resampling.NEAREST),(j*192,y)); draw.text((j*192+3,y+194),name,fill=(30,40,50))
    sheet.save(OUTPUT/'preview.png')


def main(prepare=False):
    torch.set_num_threads(4)
    expected, items, lenses, cached = prepare_inputs()
    if prepare:
        require(not PROTOCOL.parent.exists(), 'Preserve prepared specification')
        PROTOCOL.parent.mkdir(); PROTOCOL.write_text(json.dumps(expected,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'protocol_sha256':sha(PROTOCOL),'counts':expected['counts'],'new_forward_images':1593,
                          'reused_predictions':353,'held_out_forward_images':0,'optimizer_updates_locally':0,'executed':False}),flush=True)
        return
    require(PROTOCOL.is_file() and read_json(PROTOCOL)==expected and not OUTPUT.exists(), 'Frozen specification differs or evidence exists; do not restart')
    sys.path.insert(0,str(ROOT/'outputs/face_extraction_dependencies'))
    from scripts.train_face_occlusion_vm import dependency_versions
    from completion_inference import load_completion
    from face_occlusion_adapter import load_adapter
    dependencies=dependency_versions(ROOT/'outputs/face_extraction_dependencies')
    torch.set_num_threads(4)
    # Construct both models before creating output; failure cannot masquerade as a running job.
    models={'parent':load_completion(PARENT,'cpu')[0],'candidate':load_adapter(CANDIDATE,'cpu')[0]}
    OUTPUT.mkdir(); started=time.monotonic(); predictions={}; execution={}; mask_hashes={}
    partial={'complete':False,'protocol_sha256':sha(PROTOCOL),'new_forward_images':1593,'held_out_forward_images':0,
             'optimizer_constructed':False,'optimizer_updates_locally':0,'promoted':False}
    (OUTPUT/'partial.json').write_text(json.dumps(partial,indent=2)+'\n')
    for arm in ('parent','candidate'):
        chosen=items if arm=='parent' else [r for r in items if r['id'] not in cached]
        print('Starting frozen',arm,'new images',len(chosen),flush=True)
        p, report=frozen_predictions(models[arm],chosen,batch_size=8)
        predictions[arm]={**(cached if arm=='candidate' else {}),**p}; execution[arm]=report
        require(len(predictions[arm])==973,'Frozen predictions incomplete')
        for name,mask in predictions[arm].items():
            file=OUTPUT/arm/(name+'.png'); file.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(mask.astype('uint8')*255).save(file); mask_hashes[file.relative_to(OUTPUT).as_posix()]=sha(file)
        partial['completed_arms']=list(predictions); (OUTPUT/'partial.json').write_text(json.dumps(partial,indent=2)+'\n')
    scores, records, switches=scores_for(items,predictions)
    lens_scores={}
    for method in METHODS:
        recovered=sum(int((systems_for(next(r for r in items if r['id']==name),predictions)[0][method]&region).sum()) for name,region in lenses.items())
        lens_scores[method]={'recovered_pixels':recovered,'target_pixels':21853,'recovery_fraction':recovered/21853}
    training_checks={}
    for method in ('union','intersection'):
        r,p=scores[method]['real'],scores['parent']['real']; q,b=scores[method]['replay'],scores['parent']['replay']
        checks={'real_iou_gain':r['iou']>p['iou'],
                'real_no_added_errors':all(r[k]<=p[k] for k in ('visible_false_positive','empty_mask_cases','negative_false_positive_cases')),
                'replay_retention':q['iou']>=b['iou'] and all(q[k]<=b[k] for k in ('missed_fraction','visible_false_positive','empty_mask_cases','negative_false_positive_cases')),
                'lens_recovery_gain':lens_scores[method]['recovered_pixels']>lens_scores['candidate']['recovered_pixels']}
        training_checks[method]={'checks':checks,'passes':all(checks.values())}
    preview(items,predictions,expected['preview_ids'])
    require(sha(PROTOCOL)==partial['protocol_sha256'] and prepare_inputs()[0]==expected,
            'Bound code, images, masks, support or cached inputs changed during inference')
    result={**partial,'complete':True,'date':'2026-10-02','dependencies':dependencies,'torch':str(torch.__version__),
            'execution':execution,'reused_predictions':353,'mask_sha256':mask_hashes,'scores':scores,'case_counts':records,
            'lens_scores':lens_scores,'dominance_switches':switches,'training_checks':training_checks,
            'identical_rgb_label_conflicts':conflicting_inputs(items),'preview_ids':expected['preview_ids'],
            'preview_sha256':sha(OUTPUT/'preview.png'),'elapsed_seconds':time.monotonic()-started,
            'oracles_deployable':False,'training_recipe_ready':False,
            'limitations':['Training-only feasibility; no validation selection or end-to-end completion.',
                           'Oracles use known membership/targets and cannot be deployed as a single-image router.',
                           'Keeping heads frozen does not guarantee retention under learned routing.',
                           'Approximate masks and unresolved identity/pretraining overlap remain.']}
    (OUTPUT/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'complete':True,'new_forward_images':1593,'reused_predictions':353,'training_checks':training_checks,
                      'lens_scores':lens_scores,'identical_rgb_label_conflicts':len(result['identical_rgb_label_conflicts']),
                      'optimizer_updates_locally':0,'promoted':False},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--prepare',action='store_true')
    main(parser.parse_args().prepare)

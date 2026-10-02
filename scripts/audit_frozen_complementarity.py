"""Independent mask/lineage/state-fingerprint recount; no model execution."""
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'outputs/frozen_complementarity_protocol_v1/protocol.json'
OUTPUT=ROOT/'outputs/frozen_complementarity_v1'
AUDIT=ROOT/'outputs/frozen_complementarity_validation_v1'
PROTOCOL_SHA='2a42c084a969beb47613e87cca088e9fb599254335a47f76ebffcc01e72a4529'
METHODS=('parent','candidate','union','intersection','pool_oracle','dominance_oracle','pixel_oracle')


def require(condition,message):
    if not condition: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def check_execution(report,protocol):
    total=sum(protocol['counts'].values())
    require(report['complete'] is True and report['held_out_forward_images']==0
            and report['optimizer_constructed'] is False and report['optimizer_updates_locally']==0
            and report['oracles_deployable'] is False and report['training_recipe_ready'] is False
            and report['promoted'] is False and report['new_forward_images']==protocol['new_forward_images']
            and report['reused_predictions']==protocol['reused_predictions'], 'Read-only scope, inference counters or oracle status differs')
    require(set(report['execution'])=={'parent','candidate'}, 'Frozen branch missing')
    for arm,expected in (('parent',total),('candidate',total-protocol['reused_predictions'])):
        row=report['execution'][arm]; digest=row['state_before_sha256']
        require(row['forward_images']==expected and row['threshold']==protocol['threshold']==.5
                and row['model_state_unchanged'] is True and row['optimizer_constructed'] is False
                and row['optimizer_updates_locally']==0 and row['state_after_sha256']==digest
                and isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest),
                'Frozen state, threshold or actual forward count changed')
    require(sum(r['forward_images'] for r in report['execution'].values())==report['new_forward_images'], 'Planned and actual image-forward totals differ')


def independent_counts(prediction,target,valid):
    require(all(isinstance(a,np.ndarray) and a.dtype==np.bool_ and a.ndim==2
                and a.size and a.shape==target.shape for a in (prediction,target,valid))
            and valid.any() and not (target&~valid).any(), 'Matching binary prediction/target/support required')
    has=bool(target.any()); present=bool(prediction[valid].any())
    return {'tp':int(np.count_nonzero(prediction&target&valid)),
            'fp':int(np.count_nonzero(prediction&~target&valid)),
            'fn':int(np.count_nonzero(~prediction&target&valid)),
            'visible':int(np.count_nonzero(valid&~target)), 'covered_cases':int(has),
            'empty_mask_cases':int(has and not present), 'negative_cases':int(not has),
            'negative_false_positive_cases':int(not has and present),
            'ignored_positive_pixels':int(np.count_nonzero(prediction&~valid))}


def independent_systems(parent,candidate,target,valid,pool):
    require(pool in ('real','replay','reflection'), 'Unexpected nontraining pool')
    a=independent_counts(parent,target,valid); b=independent_counts(candidate,target,valid)
    better=(b['tp']>=a['tp'] and b['fp']<=a['fp'] and (b['tp']>a['tp'] or b['fp']<a['fp']))
    both=parent&candidate; either=parent|candidate
    return {'parent':parent,'candidate':candidate,'union':either,'intersection':both,
            'pool_oracle':parent if pool=='replay' else candidate,
            'dominance_oracle':candidate if better else parent,
            'pixel_oracle':(target&either)|(~target&both)},better


def aggregate(rows):
    require(rows,'Missing independent score group')
    total={k:sum(r[k] for r in rows) for k in rows[0]}
    return {**total,'cases':len(rows),'iou':total['tp']/max(1,total['tp']+total['fp']+total['fn']),
            'missed_fraction':total['fn']/max(1,total['tp']+total['fn']),
            'visible_false_positive':total['fp']/max(1,total['visible'])}


def binary(path):
    with Image.open(path) as image: a=np.array(image)
    require(a.dtype==np.uint8 and a.shape==(256,256) and np.isin(a,[0,255]).all(),'Saved mask/support is not binary 256px grayscale')
    return a.astype(bool)


def source_state_digest(path):
    payload=torch.load(path,map_location='cpu',weights_only=True); state=payload['model']; digest=hashlib.sha256()
    require(state and all(isinstance(t,torch.Tensor) and t.device.type=='cpu' and torch.isfinite(t).all() for t in state.values()),'Invalid original frozen tensor state')
    for name in sorted(state):
        value=state[name].detach().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode()); digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    torch.set_num_threads(4)
    require(not AUDIT.exists(),'Preserve completed/partial independent verification')
    require(sha(PROTOCOL)==PROTOCOL_SHA,'Prepared frozen protocol changed')
    protocol=read_json(PROTOCOL); report=read_json(OUTPUT/'results.json'); check_execution(report,protocol)
    require(report['protocol_sha256']==PROTOCOL_SHA and protocol['methods']==list(METHODS)
            and protocol['oracles_deployable'] is False and protocol['held_out_forward_images']==0
            and all(sha(ROOT/p)==h for p,h in {**protocol['input_sha256'],**protocol['code_sha256'],**protocol['reused_artifact_sha256']}.items()), 'Source/code/reuse bindings differ')
    states={}
    for arm,path in (('parent','outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth'),
                     ('candidate','outputs/downloaded_reflection_coverage/outputs/reflection_coverage_vm/reflective/epoch_42.pth')):
        states[arm]=source_state_digest(ROOT/path)
        require(states[arm]==report['execution'][arm]['state_before_sha256']==report['execution'][arm]['state_after_sha256'], 'Logged frozen state does not match actual source checkpoint tensors')
    registry_path=ROOT/'dataset/detector_supported_review_v1/manifest.json'; registry=read_json(registry_path)
    real=[r for r in registry['supported_records'] if r['split']=='train']
    require(len(real)==83 and Counter(r['kind'] for r in real)=={'covered':51,'uncovered':32},'Original training-only membership differs')
    old_protocol=read_json(ROOT/'outputs/coverage_protocol_v1/protocol.json')
    fixtures=read_json(ROOT/'outputs/reflection_coverage_data_v1/manifest.json')
    core=fixtures['core_schedule']; expected_replay=sorted({i-73 for e in core['batches'] for b in e for i in b if i>=73})
    require(len(expected_replay)==610 and all(i//10 not in core['quarantined_sources'] for i in expected_replay)
            and core['quarantined_sources']==protocol['quarantined_sources'], 'Current safe core membership changed')
    replay=torch.load(ROOT/'outputs/downloaded_coverage/outputs/coverage_training_vm/replay_pixels.pth',map_location='cpu',weights_only=True)
    pixels=torch.load(ROOT/'outputs/reflection_coverage_data_v1/pixels.pth',map_location='cpu',weights_only=True)['pixels']
    old_v2_path=ROOT/'dataset/detector_expanded_review_v2/manifest.json'
    old_v2={r['image_sha256']:r for r in read_json(old_v2_path)['records']}
    require(len(protocol['cases'])==973 and len({r['id'] for r in protocol['cases']})==973
            and Counter(r['pool'] for r in protocol['cases'])=={'real':83,'replay':610,'reflection':280},'Protocol case IDs/cohort differ')
    require([r['replay_index'] for r in protocol['cases'] if r['pool']=='replay']==expected_replay
            and [r['real_index'] for r in protocol['cases'] if r['pool']=='real']==list(range(83))
            and [r['fixture_index'] for r in protocol['cases'] if r['pool']=='reflection']==list(range(280)), 'Fixed case order differs')
    targets={}; valid_maps={}; rgb_hash_groups=defaultdict(list); collisions=[]; lenses={}; reusable={}
    for case in protocol['cases']:
        name=case['id']; pool=case['pool']
        if pool=='real':
            i=case['real_index']; row=real[i]
            require(case['source_sha256']==row['source_sha256'] and case['image_sha256']==row['image_sha256']
                    and case['new_source_index']==row.get('source_index'), 'Active real image/source membership differs')
            for key in ('image','mask','valid','source_valid'):
                p=(registry_path.parent/row[key]).resolve()
                require(p.is_relative_to(registry_path.parent) and sha(p)==row[key+'_sha256'],'Real RGB/mask/support changed')
            with Image.open(registry_path.parent/row['image']) as image: rgb=np.array(image.convert('RGB'))
            target=binary(registry_path.parent/row['mask']); valid=binary(registry_path.parent/row['valid'])
            if row.get('glare_stratum')=='strong_lens_reflection':
                lenses[name]=target&~binary(old_v2_path.parent/old_v2[row['image_sha256']]['mask'])
            elif row.get('source_index') in (171,216,348,374): lenses[name]=target.copy()
            if i<73: reusable[name]=ROOT/f'outputs/downloaded_reflection_coverage/outputs/reflection_coverage_vm/reflective/epoch_42_masks/training_real/{i:04}.png'
        elif pool=='replay':
            i=case['replay_index']; image,mask=replay[i]; source=old_protocol['sources'][i//10]
            require(case['source_sha256']==source['sha256'] and case['source_pool']==source['source'],'Replay source identity differs')
            rgb=image.permute(1,2,0).numpy(); target=mask[0].numpy().astype(bool); valid=np.ones((256,256),bool)
        else:
            i=case['fixture_index']; item=pixels[i]
            require(case['source_sha256']==fixtures['cases'][i]['source_sha256'],'Fixture source identity differs')
            rgb=item['input'].permute(1,2,0).numpy(); target=item['mask'][0].numpy().astype(bool); valid=item['valid'][0].numpy().astype(bool)
            reusable[name]=ROOT/f'outputs/downloaded_reflection_coverage/outputs/reflection_coverage_vm/reflective/epoch_42_masks/training_reflection/{i:04}.png'
        require(rgb.shape==(256,256,3) and rgb.dtype==np.uint8
                and hashlib.sha256(rgb.tobytes()).hexdigest()==case['input_byte_sha256']
                and hashlib.sha256(target.tobytes()).hexdigest()==case['target_byte_sha256']
                and hashlib.sha256(valid.tobytes()).hexdigest()==case['valid_byte_sha256']
                and int(target.sum())==case['target_pixels'] and int(valid.sum())==case['valid_pixels'], 'Input/target/support fingerprint or denominator differs')
        independent_counts(target,target,valid)
        for previous in rgb_hash_groups[case['input_byte_sha256']]:
            common=valid&valid_maps[previous]; different=(target!=targets[previous])&common
            if different.any(): collisions.append({'cases':[previous,name],'input_byte_sha256':case['input_byte_sha256'],
                'common_supervised_pixels':int(common.sum()),'conflicting_pixels':int(different.sum())})
        rgb_hash_groups[case['input_byte_sha256']].append(name); targets[name]=target; valid_maps[name]=valid
    require(len(reusable)==353 and len(lenses)==6 and sum(int(t.sum()) for t in lenses.values())==21853,
            'Reused or lens-only cohort differs')
    require(collisions==report['identical_rgb_label_conflicts'],'Identical RGB supervision conflict report differs')
    expected_files={'results.json','partial.json','preview.png'}; mask_hashes={}; branch_masks={m:{} for m in ('parent','candidate')}
    for arm in branch_masks:
        for case in protocol['cases']:
            name=case['id']; relative=f'{arm}/{name}.png'; path=OUTPUT/relative
            digest=sha(path); require(report['mask_sha256'].get(relative)==digest,'Saved head mask bytes changed')
            p=binary(path); branch_masks[arm][name]=p; expected_files.add(relative); mask_hashes[relative]=digest
            if arm=='candidate' and name in reusable:
                require(np.array_equal(p,binary(reusable[name])),'Supposed reused mask differs from prior CPU-reproduced pixels')
    require(report['mask_sha256']==mask_hashes and len(mask_hashes)==1946
            and {p.relative_to(OUTPUT).as_posix() for p in OUTPUT.rglob('*') if p.is_file()}==expected_files,
            'Extra/missing saved head evidence')
    grouped={m:defaultdict(list) for m in METHODS}; rows=[]; switches=Counter(); lens_recovery=Counter()
    for case in protocol['cases']:
        name=case['id']; target=targets[name]; valid=valid_maps[name]
        systems,dominates=independent_systems(branch_masks['parent'][name],branch_masks['candidate'][name],target,valid,case['pool'])
        counted={m:independent_counts(p,target,valid) for m,p in systems.items()}
        rows.append({'id':name,'candidate_dominates':dominates,'counts':counted})
        switches[case['pool']+'/'+('candidate' if dominates else 'parent')]+=1
        groups=(case['pool'],case['pool']+'/stratum/'+case['stratum'],case['pool']+'/source/'+case['source_pool'])
        for method in METHODS:
            for group in groups: grouped[method][group].append(counted[method])
            if name in lenses: lens_recovery[method]+=int((systems[method]&lenses[name]).sum())
    scores={m:{g:aggregate(v) for g,v in groups.items()} for m,groups in grouped.items()}
    lens_scores={m:{'recovered_pixels':lens_recovery[m],'target_pixels':21853,'recovery_fraction':lens_recovery[m]/21853} for m in METHODS}
    require(rows==report['case_counts'] and scores==report['scores'] and dict(switches)==report['dominance_switches']
            and lens_scores==report['lens_scores'], 'Independently recounted head/composition/oracle metrics differ')
    checks={}
    for method in ('union','intersection'):
        actual,parent=scores[method]['real'],scores['parent']['real']; current,base=scores[method]['replay'],scores['parent']['replay']
        flags={'real_iou_gain':actual['iou']>parent['iou'],
               'real_no_added_errors':all(actual[k]<=parent[k] for k in ('visible_false_positive','empty_mask_cases','negative_false_positive_cases')),
               'replay_retention':current['iou']>=base['iou'] and all(current[k]<=base[k] for k in ('missed_fraction','visible_false_positive','empty_mask_cases','negative_false_positive_cases')),
               'lens_recovery_gain':lens_scores[method]['recovered_pixels']>lens_scores['candidate']['recovered_pixels']}
        checks[method]={'checks':flags,'passes':all(flags.values())}
    require(checks==report['training_checks'] and report['preview_ids']==protocol['preview_ids']
            and len(set(report['preview_ids']))==10 and sha(OUTPUT/'preview.png')==report['preview_sha256'], 'Fixed decision or preview binding differs')
    with Image.open(OUTPUT/'preview.png') as preview: require(preview.size==(1152,2256),'Ten-row preview geometry differs')
    require(sha(PROTOCOL)==PROTOCOL_SHA and all(sha(ROOT/p)==h for p,h in {**protocol['input_sha256'],**protocol['code_sha256'],**protocol['reused_artifact_sha256']}.items()), 'Bound source/code changed during independent audit')
    result={'complete':True,'date':'2026-10-02','script_sha256':sha(__file__),
            'test_sha256':sha(ROOT/'tests/test_frozen_complementarity_audit.py'),'protocol_sha256':PROTOCOL_SHA,
            'result_sha256':sha(OUTPUT/'results.json'),'head_masks_recounted':1946,'diagnostic_variants_reconstructed':7,
            'case_variant_recounts':973*7,'reused_mask_pixels_equal':353,'source_state_fingerprints':states,
            'source_checkpoint_and_code_hashes_verified':True,'training_checks':checks,'lens_scores':lens_scores,
            'identical_rgb_label_conflicts':len(collisions),'model_forward_images_in_this_audit':0,
            'held_out_forward_images':0,'optimizer_updates_locally':0,'oracles_deployable':False,
            'training_recipe_ready':False,'promoted':False,
            'scope':'Independent saved-mask, support, membership, composition/bound and source-state fingerprint audit; not learned routing or model-quality promotion.'}
    AUDIT.mkdir()
    with (AUDIT/'verification.json').open('x',encoding='utf-8',newline='\n') as file: file.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='lens_scores'},indent=2),flush=True)


if __name__=='__main__': main()

"""Independently recount frozen supplemental pixels and verify matched exposure."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch


def require(condition,message):
    if not condition:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def audit_case(case,reference,style,degraded):
    require(set(case)=={'input','mask','geometry','valid'},'Unexpected cached fields')
    x=case['input'];size=x.shape[-1]
    require(x.dtype==torch.uint8 and x.shape==(3,size,size),'Expected square uint8 RGB cache')
    for key in ('mask','geometry','valid'):
        value=case[key];require(value.dtype==torch.uint8 and value.shape==(1,size,size)
                               and ((value==0)|(value==1)).all(),'Invalid binary cached regions')
    target=reference['target'].numpy();native_valid=reference['valid'].numpy();lens=reference['lens_region'].numpy()
    require(target.dtype==np.uint8 and target.shape==(size,size,3)
            and np.isin(native_valid,[0,1]).all() and np.isin(lens,[0,1]).all(),'Invalid cached reference')
    mask=case['mask'][0].numpy();geometry=case['geometry'][0].numpy();valid=case['valid'][0].numpy()
    require(not np.any(geometry & (1-lens)) and not np.any(mask & (1-valid))
            and bool(geometry.any())==(style!='clear'),'Cached geometry/foreground differs from style/support')
    expected_valid=cv2.erode(native_valid,np.ones((17,17),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=1) if degraded else native_valid
    expected_mask=(cv2.dilate(geometry,np.ones((9,9),np.uint8)) & expected_valid) if degraded else geometry
    require(np.array_equal(valid,expected_valid) and np.array_equal(mask,expected_mask),
            'Camera support/effect mask differs from declared conservative morphology')
    require(bool(mask.any())==(style!='clear'),'Camera introduced/erased an empty/covered target')
    input_rgb=x.permute(1,2,0).numpy();require(np.all(input_rgb[valid==0]==96),'Cached padded RGB differs')
    changed=np.any(input_rgb!=target,axis=2)
    if not degraded:require(not np.any(changed & (mask==0)),'Nondegraded fixture changed visible reference pixels')
    return {'hole_pixels':int(mask.sum()),'geometry_pixels':int(geometry.sum()),'valid_pixels':int(valid.sum()),
            'visible_changed_pixels':int((changed & (mask==0) & (valid==1)).sum())}


def audit(root,output):
    root=Path(root).resolve();out=(root/output).resolve();destination=out/'audit.json'
    require(out.is_relative_to(root) and not destination.exists(),'Preserve existing data audit')
    manifest_path=out/'manifest.json';manifest=read_json(manifest_path)
    require(manifest['format']=='dgp-reflection-coverage-data-v1' and manifest['training_ready'] is False
            and manifest['optimizer_updates_locally']==0,'Data stage changed')
    for category in ('input_hashes','code_hashes'):
        for name,digest in manifest[category].items():
            path=(root/name).resolve();require(path.is_relative_to(root) and sha(path)==digest,'Bound data/code input changed: '+name)
    sources=manifest['sources'];require(len(sources)==28 and len({r['sha256'] for r in sources})==28
                                       and list(sorted(manifest['source_pool_counts'].values()))==[14,14],'Source cohort differs')
    for row in sources:
        require(row['split']=='train' and row['usage']=='paired_unoccluded' and row['eyes_reviewed'] is True
                and sha(root/row['path'])==row['sha256'],'Native reviewed training source differs')
    path=out/'pixels.pth';require(sha(path)==manifest['pixel_cache_sha256'],'Frozen supplemental cache changed')
    cache=torch.load(path,map_location='cpu',weights_only=True)
    require(cache['format']=='dgp-reflection-coverage-pixels-v1' and set(cache['pixels'])==set(range(280))
            and set(cache['references'])==set(range(28)),'Cache format/membership differs')
    records=manifest['cases'];require(len(records)==280 and {r['case_id'] for r in records}==set(range(280)),'Case metadata differs')
    recounted=[]
    for row in records:
        i=row['case_id'];position=i//10;style=('clear','white_patch','white_streak','scene_reflection','blue_glare')[i%10//2]
        require(row['style']==style and row['degraded']==bool(i%2)
                and row['source_sha256']==sources[position]['sha256'],'Cached case source/style/condition differs')
        case=cache['pixels'][i]
        require(all(hashlib.sha256(t.numpy().tobytes()).hexdigest()==row['pixel_sha256'][k] for k,t in case.items()),'Raw cached tensor hash differs')
        counted=audit_case(case,cache['references'][position],style,bool(i%2))
        require(all(counted[k]==row[k] for k in ('hole_pixels','geometry_pixels','valid_pixels')),'Recount differs from metadata')
        reference=records[position*10+(i%2)]
        require(row['camera']==reference['camera'] and row['camera_seed']==reference['camera_seed'],'Camera parameters/seed not shared with control')
        recounted.append({'case_id':i,'style':style,'degraded':bool(i%2),**counted})
    protocol=read_json(root/'outputs/coverage_protocol_v1/protocol.json')
    require(sha(root/'outputs/coverage_protocol_v1/protocol.json')==manifest['protocol_sha256'],'Old protocol changed')
    original=protocol['schedules']['extended']['batches'];original=original+original[:2]
    modified=manifest['core_schedule']['batches'];quarantine=set(manifest['core_schedule']['quarantined_sources'])
    require(quarantine=={67,78,107,118,121,122,160,177} and len(modified)==12
            and all(len(e)==21 for e in modified),'Bound core budget/quarantine differs')
    replay_path=root/'outputs/downloaded_coverage/outputs/coverage_training_vm/replay_pixels.pth'
    require(sha(replay_path)==manifest['replay_sha256'],'Core replay cache changed')
    replay=torch.load(replay_path,map_location='cpu',weights_only=True);replacement_count=0
    for epoch,(before_steps,after_steps) in enumerate(zip(original,modified)):
        for step,(before,after) in enumerate(zip(before_steps,after_steps)):
            require(len(before)==len(after)==8,'Core batch shape differs')
            for a,b in zip(before,after):
                if a<73 or (a-73)//10 not in quarantine:require(a==b,'Safe/real core sample moved')
                else:
                    replacement_count+=1
                    require(b>=73 and b-73 in replay and (b-73)//10 not in quarantine,
                            'Replacement is not a source-safe frozen training case')
                    require(((a-73)%5==0)==((b-73)%5==0) and ((a-73)%10>=5)==((b-73)%10>=5),'Replacement stratum differs')
                    require(bool(replay[b-73][1].any())==((b-73)%5!=0),'Replacement target status differs')
            require(all((b-73)//10 not in quarantine for b in after if b>=73),'Quarantined core sample still exposed')
    require(replacement_count==len(manifest['core_schedule']['replacements'])==49,'Replacement count differs')
    supplemental=manifest['supplemental_schedule'];require(len(supplemental)==252,'Supplemental budget differs')
    groups=Counter();positive_ids=set();clear_ids=set()
    for step,(positive,clear) in enumerate(supplemental):
        require(positive in cache['pixels'] and clear in cache['pixels'] and positive%10//2==step%4+1
                and positive%2==(step//4)%2 and clear%10//2==0,'Scheduled supplemental stratum differs')
        groups[str((positive%10//2,positive%2))]+=1;positive_ids.add(positive);clear_ids.add(clear)
    require(len(positive_ids)==224 and len(clear_ids)==56 and max(groups.values())-min(groups.values())<=1,'Supplemental exposure product/balance differs')
    for sheet in manifest['sheets']:require(sha(out/sheet['path'])==sheet['sha256'],'Inspected preview changed')
    result={'complete':True,'script_sha256':sha(__file__),'manifest_sha256':sha(manifest_path),
            'pixel_cache_sha256':sha(path),'cases':recounted,'sources':28,'core_replacements':replacement_count,
            'unique_positive_training_cases':len(positive_ids),'unique_clear_training_cases':len(clear_ids),
            'positive_type_condition_exposures':dict(groups),'optimizer_updates_locally':0,
            'nondegraded_visible_changed_pixels':sum(r['visible_changed_pixels'] for r in recounted if not r['degraded']),
            'degraded_visible_changes':'Expected camera simulation, not introduced unmarked occlusion',
            'training_ready':False,'next':'Bound visual review and unchanged-source forward before packaging GPU runner'}
    destination.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('complete','sources','core_replacements','unique_positive_training_cases',
                                          'unique_clear_training_cases','nondegraded_visible_changed_pixels')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));parser.add_argument('--output',default='outputs/reflection_coverage_data_v1')
    args=parser.parse_args();audit(args.root,args.output)

"""Recount prepared fixture pixels and provenance without importing their renderer."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

STYLES=('clear','white_patch','white_streak','scene_reflection','blue_glare')


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_pixels(fields,style):
    require(style in STYLES, 'Unknown fixture style')
    source,target=fields['input'],fields['target']
    require(source.ndim==3 and source.shape[2]==3 and source.dtype==target.dtype==np.uint8
            and source.shape==target.shape, 'Expected matching uint8 RGB fixtures')
    mask,valid,lens=[np.asarray(fields[k]) for k in ('mask','valid','lens_region')]
    for value in (mask,valid,lens):
        require(value.shape==source.shape[:2] and value.dtype==np.uint8 and np.isin(value,[0,1]).all(),
                'Expected matching binary uint8 region/support masks')
    require(valid.any() and not np.any(mask & (1-valid)) and not np.any(lens & (1-valid)),
            'Artificial padding cannot be a supervised lens/hole')
    require(not np.any(mask & (1-lens)), 'Reflection target outside lens interiors')
    changed=np.any(source!=target,axis=2)
    require(not np.any(changed & (mask==0)), 'Visible pixels changed outside introduced reflection')
    require(np.all(source[valid==0]==96) and np.all(target[valid==0]==96), 'Padding RGB differs')
    if style=='clear':
        require(not mask.any() and not changed.any(), 'Clear lens control must have no hole or target difference')
    else:
        require(0<mask.sum()<lens.sum() and changed[mask==1].any(), 'Expected a nonempty partial reflection')
    return {'hole_pixels':int(mask.sum()),'valid_pixels':int(valid.sum()),
            'padding_pixels':int((valid==0).sum()),'lens_pixels':int(lens.sum()),
            'introduced_changed_pixels':int((changed & (mask==1)).sum()),
            'visible_changed_pixels':int((changed & (mask==0)).sum())}


def confined_file(base,name):
    require(isinstance(name,str) and name and '\\' not in name and ':' not in name
            and not PurePosixPath(name).is_absolute() and all(p not in ('..','.') for p in name.split('/')),
            'Portable confined file path required')
    path=(base/name).resolve()
    require(path.is_relative_to(base) and path.is_file(), 'Missing/escaping evidence file')
    return path


def audit(root,output):
    root=Path(root).resolve();out=(root/output).resolve()
    require(out.is_relative_to(root), 'Audit must stay within workspace')
    destination=out/'audit.json';require(not destination.exists(), 'Preserve existing fixture audit')
    registry_path=out/'registry.json';registry=json.loads(registry_path.read_text(encoding='utf-8'))
    require(registry['format']=='dgp-qualified-completion-data-v2'
            and registry['training_enabled'] is False and registry['training_recipe_ready'] is False
            and registry['optimizer_updates_locally']==0 and registry['model_inference_performed'] is False,
            'Fixture stage/training claim changed')
    for collection in ('input_hashes','code_hashes'):
        for name,digest in registry[collection].items():
            require(sha(confined_file(root,name))==digest, 'Prepared input/code bytes changed: '+name)
    records=registry['records'];require(len(records)==180, 'Original screening source count differs')
    require(all(len({r[k] for r in records})==len(records) for k in ('source_id','path','sha256')),
            'Duplicated source identity/path/bytes')
    require(dict(Counter(r['usage'] for r in records))==registry['usage_counts'], 'Qualification counts differ')
    qualified={r['source_id']:r for r in records if r['usage']=='paired_unoccluded'}
    require(set(qualified)=={0,4,101,102}, 'Reviewed prototype source membership changed')
    protocol=json.loads((root/'outputs/coverage_protocol_v1/protocol.json').read_text())
    allowed={r['path']:r['sha256'] for r in protocol['sources']}
    for row in records:
        require(row['split']=='train' and allowed.get(row['path'])==row['sha256']
                and row['detector_training_enabled'] is False and row['accepted_mask'] is None,
                'Source membership or accepted mask/training status changed')
    for identifier,row in qualified.items():
        require(sha(confined_file(root,row['path']))==row['sha256']
                and row['eyes_reviewed'] is True and row['review']['occlusion']=='none_observed',
                'Reviewed native source changed')
    cases=registry['cases'];require(len(cases)==20, 'Fixture count changed')
    expected={(i,s) for i in qualified for s in STYLES}
    require({(r['source_id'],r['style']) for r in cases}==expected, 'Fixture product differs')
    targets={};bases={};results=[];members={'registry.json'}
    for record in cases:
        identifier,style=record['source_id'],record['style'];fields={}
        require(record['split']=='train' and record['source']==qualified[identifier]['path']
                and record['source_sha256']==qualified[identifier]['sha256']
                and record['eyes_native']==qualified[identifier]['eyes']
                and record['restoration_high_resolution_reference'] is False,
                'Fixture source/task provenance differs')
        for field in ('input','target','mask','valid','base','lens_region'):
            file=record['files'][field];name=f'{field}/{identifier:03}_{style}.png'
            require(file['path']==name, 'Unexpected fixture file naming')
            path=confined_file(out,name);require(sha(path)==file['sha256'], 'Fixture bytes changed')
            with Image.open(path) as im:array=np.array(im)
            if field in ('mask','valid','lens_region'):
                require(array.dtype==np.uint8 and np.isin(array,[0,255]).all(), 'Serialized mask must be binary255')
                array=(array//255).astype(np.uint8)
            fields[field]=array;members.add(name)
        counts=audit_pixels(fields,style)
        require(counts['hole_pixels']==record['hole_pixels'] and counts['padding_pixels']==record['padding_pixels']
                and fields['input'].shape==(record['size'],record['size'],3), 'Fixture count/shape metadata differs')
        affine=np.asarray(record['affine'],np.float64);width,height=record['native_size'];size=record['size']
        require(affine.shape==(2,3) and np.isfinite(affine).all(), 'Invalid affine metadata')
        scale=(size-1)/(max(width,height)-1)
        expected_affine=np.array([[scale,0,(size-1-scale*(width-1))/2],
                                  [0,scale,(size-1-scale*(height-1))/2]])
        require(np.allclose(affine,expected_affine,rtol=0,atol=1e-12), 'Nonuniform/miscentered transform')
        eyes=np.array(record['eyes_native'])@affine[:,:2].T+affine[:,2]
        require(np.allclose(eyes,record['eyes_output'],rtol=0,atol=1e-12), 'Eye transform differs')
        for stored,field in ((targets,'target'),(bases,'base')):
            if identifier in stored:require(np.array_equal(stored[identifier],fields[field]), 'Source reference differs by style')
            else:stored[identifier]=fields[field]
        results.append({'source_id':identifier,'style':style,**counts})
    for name,digest in registry['preview_files'].items():
        require(sha(confined_file(out,name))==digest, 'Prepared preview changed');members.add(name)
    actual={p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()}
    require(actual==members, 'Missing/unexpected fixture file membership')
    result={'complete':True,'script_sha256':sha(__file__),'registry_sha256':sha(registry_path),
            'files_verified':len(members),'cases':results,'usage_counts':registry['usage_counts'],
            'training_recipe_ready':False,'optimizer_updates_locally':0,'model_inference_performed':False,
            'visual_review':'Separate assistant review required; numeric audit does not certify realism/identity/quality.',
            'limits':'Only four native low-resolution training references; procedural textures, no real transfer result. Unknown sources remain excluded.'}
    destination.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'complete':True,'files_verified':len(members),'cases_verified':len(results),
                      'visible_changed_pixels':sum(r['visible_changed_pixels'] for r in results),
                      'training_recipe_ready':False},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--output',default='outputs/qualified_reflection_v1')
    args=parser.parse_args();audit(args.root,args.output)

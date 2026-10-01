"""Prepare source-qualified reflection fixtures; no model or optimizer execution."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from completion_data_v2 import FORMAT, REFLECTION_STYLES, prepare_case, require

PROTOCOL_SHA='46c438f6e72e9bd4709102dd8147dc926fff84de0e7e8a89f3bee1c22a9590fd'
MANIFEST_SHA='e36ce5cf04c858d61885c2a0187d0182099ada9852eb03d21d645f5bdbb3ea18'
NATIVE_REVIEWS={
    0: ([[42.,93.],[110.,90.]], 'Both eyes and lower face visible in native154x196; limited sharpness, no added detail claimed.'),
    4: ([[47.,101.],[108.,105.]], 'Both eyes and lower face visible in native159x215; off-center gaze retained.'),
    101: ([[49.,60.],[79.,60.]], 'Both eyes visible in native128x128; the adjacent toy does not overlap facial features.'),
    102: ([[49.,62.],[80.,62.]], 'Both eyes and lower face visible in native128x128; skin markings remain unchanged.'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def qualification_records(cohort,screening):
    rows=cohort['records']
    require(all(len({r[k] for r in rows})==len(rows) for k in ('source_id','path','sha256')),
            'Qualification sources must have unique IDs, paths and bytes')
    screened={r['source_id']:r for r in screening['records']}
    require(set(NATIVE_REVIEWS).issubset({r['source_id'] for r in rows}), 'Native prototype source missing')
    result=[]
    for source in rows:
        identifier=source['source_id'];previous=screened.get(identifier)
        record={**source,'split':'train','usage':'pending','eyes':None,'eyes_reviewed':False,
                'review':{'method':'not_native_qualified','occlusion':'unknown',
                          'reference_quality':'unknown','rationale':'Contact screen is insufficient for paired-source eligibility.'},
                'detector_training_enabled':False,'accepted_mask':None}
        if previous:
            record['prior_scope_screen']=previous['category']
            record['review']['rationale']=previous['rationale']
            if previous['category']=='likely_intrinsic_occlusion':
                record['usage']='unpaired_occlusion'
                record['review'].update(method='assistant_native_screen',occlusion='likely_observed')
        if identifier in NATIVE_REVIEWS:
            require(previous is None or previous['category'] in ('clear_lens_control','not_lens_glare'),
                    'Prototype cannot overrule a prior intrinsic/uncertain source screen')
            eyes,rationale=NATIVE_REVIEWS[identifier]
            record.update(usage='paired_unoccluded',eyes=eyes,eyes_reviewed=True,
                          review={'method':'assistant_native_screen','occlusion':'none_observed',
                                  'reference_quality':'native_low_resolution','rationale':rationale})
        result.append(record)
    return result


def prepare(root,output,size=256,seed=42):
    root=Path(root).resolve();out=(root/output).resolve()
    require(out.is_relative_to(root) and not out.exists(), 'Choose a new workspace output; existing evidence is preserved')
    scope=root/'outputs/clear_replay_scope_v1'
    cohort=read_json(scope/'cohort.json');screening=read_json(scope/'source_screening.json')
    require(screening['cohort_sha256']==sha(scope/'cohort.json') and screening['source_screen_complete'] is True,
            'Complete source screen and exact cohort required')
    protocol_path=root/'outputs/coverage_protocol_v1/protocol.json'
    require(sha(protocol_path)==cohort['protocol_sha256']==PROTOCOL_SHA, 'Training source protocol changed')
    protocol=read_json(protocol_path)
    training={s['path'].replace('\\','/'):s['sha256'] for s in protocol['sources']}
    require(len(training)==len(protocol['sources'])==200, 'Expected200 unique original training sources')
    real_path=root/'dataset/detector_glare_review_v3/manifest.json'
    require(sha(real_path)==MANIFEST_SHA, 'Reviewed V3 manifest changed')
    real=read_json(real_path)['records']
    benchmark_path=root/'outputs/downloaded_completion_benchmark/outputs/completion_pretrained_vm/manifest.json'
    benchmark=read_json(benchmark_path)['cases']
    excluded={c['source_sha256'] for c in benchmark}
    excluded.update(r[k] for r in real for k in ('source_sha256','image_sha256'))
    require(not (set(training.values()) & excluded), 'Training protocol overlaps fixed benchmark/reviewed inputs')
    for source in cohort['records']:
        require(training.get(source['path'])==source['sha256'], 'Screened source not in exact training protocol')
    preview_path=root/'outputs/qualified_source_review_v1/preview_cohort.json'
    preview_cohort=read_json(preview_path)
    expected_preview=[]
    for pool in sorted({r['source_pool'] for r in cohort['records']}):
        expected_preview.extend([r for r in cohort['records'] if r['source_pool']==pool][:6])
    require([r['source_id'] for r in preview_cohort['records']]==[r['source_id'] for r in expected_preview]
            and all(training.get(r['path'])==r['sha256'] for r in preview_cohort['records'])
            and set(NATIVE_REVIEWS).issubset({r['source_id'] for r in preview_cohort['records']}),
            'Native review prototypes differ from the declared training-only source cohort')
    old_data=root/'completion_data.py'
    require(sha(old_data)==screening['code_reference_sha256'], 'Existing dataset code changed since source screen')
    records=qualification_records(cohort,screening)
    qualified=[r for r in records if r['usage']=='paired_unoccluded']
    require(len(qualified)==4, 'Fixed source-only prototype size changed')
    # Generate all cases before creating an evidence directory: invalid anchors
    # or provenance cannot leave a misleading partially prepared registry.
    cases=[(row,style,prepare_case(root,row,training,excluded,style,seed,size))
           for row in qualified for style in ('clear',*REFLECTION_STYLES)]
    out.mkdir(parents=True)
    for name in ('input','target','mask','valid','base','lens_region'):(out/name).mkdir()
    metadata=[]
    for row,style,case in cases:
        name=f'{row["source_id"]:03}_{style}.png'
        files={}
        for field in ('input','target','mask','valid','base','lens_region'):
            array=case[field]*(255 if field in ('mask','valid','lens_region') else 1)
            path=out/field/name;Image.fromarray(array).save(path)
            files[field]={'path':f'{field}/{name}','sha256':sha(path)}
        metadata.append({**case['metadata'],'files':files})
    fixture_by_key={(row['source_id'],style):case for row,style,case in cases}
    # A fixed training-only grid, not a metric/error-ranked cohort.
    preview=Image.new('RGB',(size*6,24+(size+24)*len(qualified)),'white')
    draw=ImageDraw.Draw(preview)
    for col,label in enumerate(('base','clear',*REFLECTION_STYLES)):
        draw.text((col*size+3,4),label,fill='black')
    geometry=Image.new('RGB',(size*3,24+(size+24)*len(qualified)),'white')
    gd=ImageDraw.Draw(geometry)
    for col,label in enumerate(('existing stretch','uniform scale','source support')):
        gd.text((col*size+3,4),label,fill='black')
    for line,row in enumerate(qualified):
        yy=24+line*(size+24)
        for col,style in enumerate(('base','clear',*REFLECTION_STYLES)):
            case=fixture_by_key[row['source_id'],'clear' if style=='base' else style]
            tile=case['base'] if style=='base' else case['input']
            preview.paste(Image.fromarray(tile),(col*size,yy))
            draw.text((col*size+3,yy+size+2),f'source{row["source_id"]}',fill='black')
        native=cv2.imread(str(root/row['path']))
        stretched=cv2.cvtColor(cv2.resize(native,(size,size)),cv2.COLOR_BGR2RGB)
        base=fixture_by_key[row['source_id'],'clear']
        for col,array in enumerate((stretched,base['base'],np.repeat((base['valid']*255)[:,:,None],3,axis=2))):
            geometry.paste(Image.fromarray(array),(col*size,yy))
            gd.text((col*size+3,yy+size+2),f'source{row["source_id"]}',fill='black')
    preview.save(out/'reflection_preview.png');geometry.save(out/'geometry_preview.png')
    inputs={str(p.relative_to(root)).replace('\\','/'):sha(p) for p in
            (scope/'cohort.json',scope/'source_screening.json',protocol_path,real_path,benchmark_path,old_data,preview_path)}
    registry={'format':FORMAT,'stage':'source-qualified prototype; augmentation review pending',
              'records':records,'usage_counts':dict(Counter(r['usage'] for r in records)),
              'input_hashes':inputs,'code_hashes':{p:sha(root/p) for p in
                 ('completion_data_v2.py','scripts/prepare_qualified_reflection_review.py')},
              'accepted_masks_created':0,'training_enabled':False,'optimizer_updates_locally':0,
              'source_policy':'Reviewed uncovered references only; unknown/obscured faces stay out of paired examples.',
              'review_level':'Assistant native source screening and manually proposed eye centers; no independent expert adjudication.',
              'selection':'Four eligible prototypes from first six clear training source IDs per pool; no validation/test/error ranking',
              'source_preview_cohort_sha256':sha(preview_path),
              'unqualified_sources':'All pending sources remain excluded even if contact-screened or native-viewed. No procedural zero target or absence of a screen flag grants eligibility.',
              'high_resolution_ground_truth':False,'cases':metadata,
              'preview_files':{p:sha(out/p) for p in ('reflection_preview.png','geometry_preview.png')},
              'training_recipe_ready':False,'model_inference_performed':False,
              'scope':'Training source-only prototype. Original cached masks, benchmark, teachers, splits, gates and application retained.',
              'next':'Inspect eye/frame placement and patch masks; independently recount fixtures before matched GPU recipe.'}
    require(all(sha(root/p)==digest for p,digest in inputs.items()), 'Frozen input changed during preparation')
    (out/'registry.json').write_text(json.dumps(registry,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(out),'usage_counts':registry['usage_counts'],
                      'procedural_cases':len(metadata),'training_recipe_ready':False},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(ROOT))
    parser.add_argument('--output',default='outputs/qualified_reflection_v1')
    parser.add_argument('--size',type=int,default=256)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args();prepare(args.root,args.output,args.size,args.seed)

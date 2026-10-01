"""Freeze native-reviewed reflection/camera pixels and a matched training schedule."""
import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from completion_data_v2 import prepare_case,REFLECTION_STYLES,require
from reflection_coverage import camera_fixture,supplemental_schedule,sanitize_schedule

PROTOCOL_SHA='46c438f6e72e9bd4709102dd8147dc926fff84de0e7e8a89f3bee1c22a9590fd'
REPLAY_SHA='ad6fd64aed4f773c968cac7748be0fd99a7c8679d1019e32fc64a03b13662dc7'
QUALIFIED_ADDITIONS=(7,9,10,13,15,17,18,19,20,21,22,26,111,113,114,116,117,119,120,123,124,128,131,132)
QUARANTINE=(67,78,107,118,121,122,160,177)
DEFERRED={
    6:'Severe native blur; eye detail insufficient for this prototype.',
    11:'Underexposure and limited eye detail; keep pending.',
    12:'Downward gaze/closed lids; exclude from the current two-visible-eye fixture.',
    14:'Severe native blur; keep pending.',
    16:'Strong profile; equal-lens fixture is unsuitable.',
    24:'Shadowed squint/oblique pose; keep pending for this fixture.',
    28:'Severe native blur; keep pending.',
    29:'Oblique pose; equal-lens fixture is unsuitable.',
    30:'Oblique pose; keep pending for this fixture.',
    108:'Strong profile; equal-lens fixture is unsuitable.',
    109:'Oblique pose; keep pending for this fixture.',
    112:'Strong colored illumination; defer this initial prototype.',
    115:'Existing transparent eyewear; adding a second frame is unsuitable. No covered label accepted.',
    118:'Native hand lies over the side/ear boundary: likely intrinsic covering candidate; mask not adjudicated.',
    121:'Prior lens-reflection boundary ambiguity remains unresolved.',
    122:'Existing eyewear/reflection and oblique pose need scope review; no covered label accepted.',
    125:'Oblique pose; equal-lens fixture deferred.',
    126:'Oblique pose; equal-lens fixture deferred.',
    127:'Existing eyewear; do not add a second frame. No covered label accepted.',
    130:'Oblique pose; equal-lens fixture deferred.',
}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def qualify_sources(prior,proposals):
    require(proposals['training_enabled'] is False and proposals['native_review_complete'] is False,
            'Require original unapproved eye proposals')
    records=copy.deepcopy(prior['records']);by_id={r['source_id']:r for r in records}
    require(len(by_id)==len(records),'Duplicate original source ID')
    declared=set(QUALIFIED_ADDITIONS)|set(DEFERRED)
    require({r['source_id'] for r in proposals['records']}==declared and len(proposals['records'])==len(declared),
            'Fixed native review cohort differs')
    for proposal in proposals['records']:
        i=proposal['source_id'];row=by_id.get(i)
        require(row is not None and row['path']==proposal['path'] and row['sha256']==proposal['sha256']
                and proposal.get('eyes_reviewed') is False and proposal['training_enabled'] is False,
                'Native proposal changed membership or was automatically approved')
        row.update(detector_training_enabled=False,accepted_mask=None)
        if i in QUALIFIED_ADDITIONS:
            eyes=proposal.get('eyes_native')
            require(proposal['decision']=='proposed' and np.asarray(eyes).shape==(2,2)
                    and np.isfinite(eyes).all(), 'Missing/unreliable native eye proposal')
            row.update(usage='paired_unoccluded',eyes=eyes,eyes_reviewed=True,
                       review={'method':'assistant_native_screen','occlusion':'none_observed',
                               'reference_quality':'native_low_resolution',
                               'rationale':'Native-aspect source and eye proposal inspected: two visible eyes; no observed covering in the fixture features. Limited native resolution retained.'},
                       eye_proposal_review='Existing detector proposals visually checked; no automatic confidence-based eligibility')
        else:
            row.update(eyes=None,eyes_reviewed=False,usage='unpaired_occlusion' if i==118 else 'pending',
                       review={'method':'assistant_native_screen','occlusion':'likely_observed' if i==118 else 'not_qualified',
                               'reference_quality':'native_low_resolution','rationale':DEFERRED[i]})
    require({r['source_id'] for r in records if r['usage']=='paired_unoccluded'}==set(QUALIFIED_ADDITIONS)|{0,4,101,102},
            'Native reviewed prototype membership differs')
    return records


def prepare(root,output):
    root=Path(root).resolve();out=(root/output).resolve()
    require(out.is_relative_to(root) and not out.exists(),'Choose a fresh evidence directory')
    previous=root/'outputs/qualified_reflection_v1';prior=read_json(previous/'registry.json')
    prior_audit=read_json(previous/'audit.json');visual=read_json(previous/'visual_review.json')
    require(sha(previous/'registry.json')==prior_audit['registry_sha256']==visual['registry_sha256']
            and sha(previous/'audit.json')==visual['audit_sha256'] and prior_audit['complete'] is True,
            'Verified source prototype lineage required')
    proposal_path=root/'outputs/reflection_source_cohort_v2/proposals.json';proposals=read_json(proposal_path)
    require(sha(root/'scripts/prepare_reflection_source_cohort.py')==proposals['script_sha256']
            and sha(root/'completion_data_v2.py')==proposals['geometry_module_sha256']
            and all(sha(root/'outputs/reflection_source_cohort_v2'/s['path'])==s['sha256'] for s in proposals['sheets']),
            'Inspected eye proposal/sheet code bytes changed')
    protocol_path=root/'outputs/coverage_protocol_v1/protocol.json'
    require(sha(protocol_path)==PROTOCOL_SHA,'Fixed original protocol changed');protocol=read_json(protocol_path)
    for name,digest in protocol['input_hashes'].items():require(sha(root/name)==digest,'Original source exclusion input changed')
    allowed={r['path']:r['sha256'] for r in protocol['sources']}
    benchmark=read_json(root/'outputs/downloaded_completion_benchmark/outputs/completion_pretrained_vm/manifest.json')['cases']
    reviewed=read_json(root/'dataset/detector_glare_review_v3/manifest.json')['records']
    excluded={r['source_sha256'] for r in benchmark}|{r[k] for r in reviewed for k in ('image_sha256','source_sha256')}
    require(not set(allowed.values())&excluded,'Fixed sources overlap prior benchmark/reviewed data')
    records=qualify_sources(prior,proposals);qualified=sorted((r for r in records if r['usage']=='paired_unoccluded'),key=lambda r:r['source_id'])
    pools=Counter(r['source_pool'] for r in qualified);require(list(sorted(pools.values()))==[14,14],'Expected balanced source-pool prototype')
    replay_path=root/'outputs/downloaded_coverage/outputs/coverage_training_vm/replay_pixels.pth'
    require(sha(replay_path)==REPLAY_SHA,'Fixed replay bytes changed');replay=torch.load(replay_path,map_location='cpu',weights_only=True)
    # Twelve epochs permit every28-reference ×4-style ×2-condition combination
    # to appear at least once. The shared tail repeats the first two old epochs.
    original=copy.deepcopy(protocol['schedules']['extended']['batches'])
    require(len(original)==10 and all(len(e)==21 for e in original),'Expected original210-step core')
    original.extend(copy.deepcopy(original[:2]))
    core=sanitize_schedule(original,set(replay),QUARANTINE,seed=42)
    supplemental=supplemental_schedule(len(qualified),252,42)
    pixels={};references={};case_rows=[]
    for position,row in enumerate(qualified):
        for style_index,style in enumerate(('clear',*REFLECTION_STYLES)):
            base=prepare_case(root,row,allowed,excluded,style,42,256)
            if style_index==0:
                references[position]={key:torch.from_numpy(base[key].copy()) for key in ('target','base','lens_region','valid')}
            for condition in range(2):
                camera_seed=(42+int(row['sha256'][:8],16)+condition*2000003)%(2**32)
                case=camera_fixture(base['input'],base['mask'],base['valid'],camera_seed,bool(condition))
                i=position*10+style_index*2+condition
                require(bool(case['mask'].any())==(style!='clear'),'Camera changed empty/covered target status')
                pixels[i]={'input':torch.from_numpy(case['input'].copy()).permute(2,0,1).contiguous(),
                           'mask':torch.from_numpy(case['mask'][None].copy()),
                           'geometry':torch.from_numpy(base['mask'][None].copy()),
                           'valid':torch.from_numpy(case['valid'][None].copy())}
                case_rows.append({**base['metadata'],'case_id':i,'source_position':position,'degraded':bool(condition),
                                  'camera_seed':camera_seed,'camera':case['camera'],
                                  'hole_pixels':int(case['mask'].sum()),'geometry_pixels':int(base['mask'].sum()),
                                  'valid_pixels':int(case['valid'].sum()),
                                  'pixel_sha256':{k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in pixels[i].items()}})
    out.mkdir(parents=True);cache_path=out/'pixels.pth'
    torch.save({'format':'dgp-reflection-coverage-pixels-v1','pixels':pixels,'references':references},cache_path)
    # Source order, without prediction/error ranking; four full style sheets.
    sheets=[]
    for start in range(0,len(qualified),7):
        subset=qualified[start:start+7];sheet=Image.new('RGB',(1536,24+len(subset)*280),'white');draw=ImageDraw.Draw(sheet)
        for col,label in enumerate(('base','clear',*REFLECTION_STYLES)):draw.text((col*256+3,4),label,fill='black')
        for line,row in enumerate(subset):
            position=start+line;yy=24+line*280
            for col in range(6):
                tile=references[position]['base'].numpy() if col==0 else pixels[position*10+(col-1)*2]['input'].permute(1,2,0).numpy()
                sheet.paste(Image.fromarray(tile),(col*256,yy));draw.text((col*256+3,yy+258),f'source{row["source_id"]}',fill='black')
        name=f'reflection_preview_{start//7:02}.png';sheet.save(out/name)
        sheets.append({'path':name,'sha256':sha(out/name),'source_ids':[r['source_id'] for r in subset]})
    # Fixed first/last source in each pool, clear camera/known mask comparison.
    diagnostic_positions=[0,13,14,27]
    sheet=Image.new('RGB',(1024,24+4*280),'white');draw=ImageDraw.Draw(sheet)
    for col,label in enumerate(('target','degraded clear','degraded reflection','effect mask')):draw.text((col*256+3,4),label,fill='black')
    for line,position in enumerate(diagnostic_positions):
        yy=24+line*280;items=[references[position]['target'].numpy(),pixels[position*10+1]['input'].permute(1,2,0).numpy(),
                            pixels[position*10+9]['input'].permute(1,2,0).numpy(),np.repeat((pixels[position*10+9]['mask'][0].numpy()*255)[:,:,None],3,axis=2)]
        for col,array in enumerate(items):sheet.paste(Image.fromarray(array),(col*256,yy));draw.text((col*256+3,yy+258),f'source{qualified[position]["source_id"]}',fill='black')
    name='camera_preview.png';sheet.save(out/name);sheets.append({'path':name,'sha256':sha(out/name),'source_ids':[qualified[i]['source_id'] for i in diagnostic_positions]})
    manifest={'format':'dgp-reflection-coverage-data-v1','records':records,'sources':qualified,'cases':case_rows,
              'usage_counts':dict(Counter(r['usage'] for r in records)),'source_pool_counts':dict(pools),
              'protocol_sha256':PROTOCOL_SHA,'replay_sha256':REPLAY_SHA,
              'prior_registry_sha256':sha(previous/'registry.json'),'prior_audit_sha256':sha(previous/'audit.json'),
              'proposal_sha256':sha(proposal_path),'pixel_cache_sha256':sha(cache_path),'sheets':sheets,
              'core_schedule':core,'supplemental_schedule':supplemental,'epochs_per_arm':12,'steps_per_epoch':21,
              'source_membership':'Original training source-ID pool only; no held-out example decoded or trained',
              'native_review':'All44 fixed native-aspect proposal photos inspected.24 additions admitted; others remain pending/unpaired. Assistant review, not independent expert adjudication.',
              'quarantine_policy':'Eight uncertain/likely intrinsically obscured source IDs omitted from core exposure in both arms. Cached pixels/labels remain unchanged.',
              'supplemental_policy':'Control uses same-source/same-camera clear replacement for the positive slot; reflective arm uses the registered reflection. Second slot is identical clear input in both arms.',
              'support_policy':'All supplemental supervised reductions ignore valid=0. Core supervision/teacher terms retain their original8-example definitions.',
              'teacher_policy':'Original parent KL on core synthetic replay only; no parent pseudo-labels on new lens fixtures.',
              'input_hashes':{str(p.relative_to(root)).replace('\\','/'):sha(p) for p in
                              (previous/'registry.json',previous/'audit.json',previous/'visual_review.json',proposal_path,protocol_path)},
              'code_hashes':{p:sha(root/p) for p in ('completion_data_v2.py','reflection_coverage.py','scripts/prepare_reflection_coverage_data.py')},
              'training_ready':False,'optimizer_updates_locally':0,'model_inference':'Keypoint proposals only; no occlusion detector fitting/inference',
              'limits':'Low-resolution references and simplified frame/reflection/camera simulations; transfer, identity separation and externally pretrained overlap remain unproven.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'sources':len(qualified),'pools':dict(pools),'cases':len(case_rows),
                      'shared_core_replacements':len(core['replacements']),'updates_per_arm':252,
                      'training_ready':False,'output':str(out)},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(ROOT));parser.add_argument('--output',default='outputs/reflection_coverage_data_v1')
    args=parser.parse_args();prepare(args.root,args.output)

"""Prepare forty mild paired DEV inputs and an anonymous input-only packet.

No model, HTTP, optimizer, gradients, training, VM or visual admission runs here.
Every destination is new and exclusive. Prior study/provenance stays unchanged.
"""
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import random
import stat
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as PILLOW_VERSION
import psutil
import scipy
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'outputs/codeformer_u96_context_study_v1'
OUT = ROOT/'outputs/codeformer_u96_mild_context_v1'
PACK = ROOT/'scratch/u96_mild_input_admission_v1'
SOURCE_BUNDLE = ROOT/'outputs/cctv_codeformer_adaptation_v2_vm'
REVIEW_SHA = 'a1ee7e6b62920e408c16b3b7a962bbc75074bc57f906a022d684bb7fd6e0ba50'
CANDIDATE_SHA = 'faec2e7f00960ba76ccc4751ba0dfdcda14a2c6d3e6f83b2cb55e27bf7bdeede'
SOURCE_SHA = '55b892ee5d55a123ea6cb55f84f322824faaec956f59168e443ff94b93337cde'
CRITERIA = ROOT/'outputs/codeformer_filter_pairs_v1_preparation/input_only_review_criteria_v1.json'
LIMITS = {'wall_seconds':120, 'reserve_bytes':4*1024**3,
          'generated_bytes':64*1024**2, 'reference_count':8, 'input_count':40,
          'anonymous_degraded_count':32, 'input_sheets':8}
ROLE_SIGMA = [('clear',None), ('global_gaussian_1p2',1.2),
              ('global_gaussian_1p8',1.8), ('local_gaussian_1p2',1.2),
              ('local_gaussian_1p8',1.8)]
START = time.monotonic()
WRITTEN = 0


def sha(raw):return hashlib.sha256(raw).hexdigest()


def guard():
    assert time.monotonic()-START < LIMITS['wall_seconds'], 'Preparation wall cap'
    assert psutil.disk_usage(str(ROOT)).free >= LIMITS['reserve_bytes'], 'Disk reserve'
    assert WRITTEN < LIMITS['generated_bytes'], 'Generated payload cap'


def safe(path):
    path = Path(path).absolute()
    assert path.is_relative_to(ROOT) and path.resolve().is_relative_to(ROOT)
    for item in [path,*path.parents]:
        if item == ROOT.parent:break
        if item.exists():
            info=item.lstat()
            assert not stat.S_ISLNK(info.st_mode) and not getattr(info,'st_file_attributes',0)&1024
    return path


def pin(path):
    path=safe(path); raw=path.read_bytes()
    return {'path':path.relative_to(ROOT).as_posix(),'bytes':len(raw),'sha256':sha(raw)}


def bound(row):
    path=safe(ROOT/row['path']);raw=path.read_bytes()
    assert len(raw)==row['bytes'] and sha(raw)==row['sha256'], row['path']
    return raw


def put(path,raw):
    global WRITTEN
    guard();assert WRITTEN+len(raw) < LIMITS['generated_bytes']
    with path.open('xb') as stream:stream.write(raw)
    WRITTEN+=len(raw);guard()


def dump(path,value):
    put(path,(json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False)+'\n').encode('utf-8'))


def png(rgb):
    stream=io.BytesIO();Image.fromarray(rgb).save(stream,format='PNG');return stream.getvalue()


def decode(raw,size):
    with Image.open(io.BytesIO(raw)) as image:
        assert image.mode=='RGB' and image.size==size and getattr(image,'n_frames',1)==1
        return np.asarray(image).copy()


def blur(image,sigma):
    return np.rint(gaussian_filter(image.astype(np.float64),(sigma,sigma,0),
                                  mode='reflect',truncate=3.)).clip(0,255).astype(np.uint8)


def main():
    guard()
    assert not OUT.exists() and not PACK.exists(), 'Never overwrite an earlier/partial packet'
    ast.parse(Path(__file__).read_bytes(),filename=__file__)
    old_review_path=OLD/'root_input_review_v1.json'
    old_manifest_path=OLD/'candidate_manifest_v1.json'
    assert pin(old_review_path)['sha256']==REVIEW_SHA
    assert pin(old_manifest_path)['sha256']==CANDIDATE_SHA
    old_review=json.loads(old_review_path.read_bytes())
    old_manifest=json.loads(old_manifest_path.read_bytes())
    assert old_review['candidate_manifest']==pin(old_manifest_path)
    ids=old_review['selected_reference_ids'];assert len(ids)==len(set(ids))==8
    old_sources={r['reference_id']:r for r in old_manifest['sources']}
    original_refs_path=ROOT/old_manifest['reference_manifest']['path']
    assert pin(original_refs_path)==old_manifest['reference_manifest']
    assert pin(original_refs_path)['sha256']==SOURCE_SHA
    refs=json.loads(original_refs_path.read_bytes())['references']
    refs_by_id={r['id']:r for r in refs};assert len(refs_by_id)==len(refs)
    train=[r for r in refs if r['historical_role']=='train']
    train_ids={r['id'] for r in train};train_hashes={r['source_sha256'] for r in train}
    train_photos={r['attribution']['photo_url'] for r in train}
    membership_path=ROOT/old_manifest['historical_membership_manifest']['path']
    assert pin(membership_path)==old_manifest['historical_membership_manifest']
    membership=json.loads(membership_path.read_bytes())
    assert set(ids).issubset(set(membership['validation_reference_ids']))
    assert not set(ids)&set(membership['training_reference_ids'])
    criteria=json.loads(CRITERIA.read_bytes())
    required={'readable','borderline','insufficient','out_of_scope'}
    assert set(criteria['decisions'])==required
    prior_boundaries=[pin(old_review_path),pin(old_manifest_path),pin(original_refs_path),
                      pin(membership_path),pin(CRITERIA)]
    # Finish all exact source and membership checks before output construction.
    checked=[]
    for reference_id in ids:
        guard();old=old_sources[reference_id];ref=refs_by_id[reference_id]
        assert old['historical_role']=='validation' and old['study_role']=='DEV'
        assert ref['historical_role']=='validation' and ref['use_for_optimization'] is False
        assert ref['decision']=='accept_clean_restoration' and reference_id not in train_ids
        assert ref['source_sha256'] not in train_hashes
        assert ref['attribution']['photo_url'] not in train_photos
        assert old['attribution']==ref['attribution']
        assert all(ref['attribution'][k] for k in ['photo_url','author','license','license_url'])
        source=safe(SOURCE_BUNDLE/ref['source_path'])
        target=safe(SOURCE_BUNDLE/ref['existing_target_path'])
        assert pin(source)==old['source1024'] and pin(target)==old['target256_original']
        assert pin(source)['sha256']==ref['source_sha256']
        assert pin(target)['sha256']==ref['existing_target_sha256']
        source_bytes=bound(old['source1024']);target_bytes=bound(old['target256_original'])
        source_rgb=decode(source_bytes,(1024,1024));target_rgb=decode(target_bytes,(256,256))
        assert sha(source_rgb.tobytes())==old['source_RGB8_sha256']
        assert sha(target_rgb.tobytes())==old['target_RGB8_sha256']
        assert bound(old['target256_saved'])==target_bytes
        with Image.open(io.BytesIO(source_bytes)) as image:
            expected=np.asarray(image.resize((256,256),Image.Resampling.LANCZOS))
        assert np.array_equal(expected,target_rgb),reference_id
        checked.append((reference_id,old,ref,target_bytes,target_rgb))
    OUT.mkdir();(OUT/'inputs').mkdir();(OUT/'targets').mkdir();PACK.mkdir()
    sources=[];cases=[]
    for reference_id,old,ref,target_bytes,target in checked:
        target_path=OUT/'targets'/f'{reference_id}.png';put(target_path,target_bytes)
        sources.append({'reference_id':reference_id,'historical_role':'validation','study_role':'DEV',
                        'source1024':old['source1024'],'target256_original':old['target256_original'],
                        'prior_saved_target':old['target256_saved'],'target256_saved':pin(target_path),
                        'source_RGB8_sha256':old['source_RGB8_sha256'],
                        'target_RGB8_sha256':old['target_RGB8_sha256'],
                        'source_to_target_RGB1024_LANCZOS256_exact':True,
                        'attribution':ref['attribution'],'attribution_audit':'Exact retained terms snapshot; not newly fetched licensing/identity certification',
                        'official_FFHQ_category':ref['official_category'],
                        'official_category_does_not_change_project_DEV_membership':True,
                        'original_source_path':ref['original_source_path'],
                        'historical_source_decision':ref['decision'],
                        'historical_source_reviewed_at_original_resolution':ref['source_reviewed_at_original_resolution'],
                        'new_source_visual_review_performed':False,'optimization_permitted':False})
        filtered={sigma:blur(target,sigma) for sigma in [1.2,1.8]}
        for role,sigma in ROLE_SIGMA:
            guard();image=target.copy()
            if role.startswith('global'):image=filtered[sigma].copy()
            if role.startswith('local'):image[64:192,64:192]=filtered[sigma][64:192,64:192]
            path=OUT/'inputs'/f'{reference_id}__{role}.png'
            put(path,target_bytes if role=='clear' else png(image))
            assert np.array_equal(decode(path.read_bytes(),(256,256)),image)
            if role.startswith('local'):
                outside=np.ones((256,256),dtype=bool);outside[64:192,64:192]=False
                assert np.array_equal(image[outside],target[outside])
                assert np.array_equal(image[64:192,64:192],filtered[sigma][64:192,64:192])
            cases.append({'id':f'{reference_id}__{role}','reference_id':reference_id,'role':role,
                          'split':'DEV','input':pin(path),'input_RGB8_sha256':sha(image.tobytes()),
                          'target':pin(target_path),'paired_clean_target':True,
                          'Gaussian_sigma':sigma,'local_outside_corruption_ROI_exact':True if role.startswith('local') else None,
                          'input_only_readability':'PENDING','pose_review':'PENDING',
                          'protection_selected':False,'selected_for_inference':False,
                          'optimization_permitted':False})
    assert len(cases)==40 and len(sources)==8
    degraded=[r for r in cases if r['role']!='clear'];assert len(degraded)==32
    random.Random(2026102104).shuffle(degraded)
    mapping=[]
    for index,case in enumerate(degraded,1):
        path=PACK/f'input_{index:02d}.png';put(path,bound(case['input']))
        mapping.append({'anonymous_id':f'I{index:02d}','id':case['id'],
                        'reference_id':case['reference_id'],'role':case['role'],
                        'original_input':case['input'],'anonymous_input':pin(path)})
    font_path=Path('C:/Windows/Fonts/arial.ttf');assert font_path.is_file()
    font=ImageFont.truetype(str(font_path),16)
    sheets=[]
    for start in range(0,32,4):
        sheet=Image.new('RGB',(1088,296),'#e6e2dc');draw=ImageDraw.Draw(sheet)
        for slot,item in enumerate(mapping[start:start+4]):
            draw.text((slot*272+8,8),item['anonymous_id'],font=font,fill='black')
            image=decode(bound(item['anonymous_input']),(256,256))
            sheet.paste(Image.fromarray(image),(slot*272+8,32))
            assert np.array_equal(np.asarray(sheet)[32:288,slot*272+8:slot*272+264],image)
        path=PACK/f'inputs_{start//4+1:02d}.png';stream=io.BytesIO();sheet.save(stream,format='PNG')
        put(path,stream.getvalue());sheets.append({'file':pin(path),'anonymous_ids':[r['anonymous_id'] for r in mapping[start:start+4]],'exact256_pixel_pastes':True})
    rules={'purpose':'Prospective degraded-input-only admission before any model output; no source/target counterpart views.',
           'observation':criteria['observation'],'decisions':criteria['decisions'],
           'prohibited_decision_evidence':criteria['prohibited_decision_evidence'],
           'reviewer_access_limit':'Only this scratch/u96_mild_input_admission_v1 folder. Do not read other project files, maps, targets, outputs or parent history.',
           'reviewer_must_not_infer_identity_or_ethnicity':True,
           'clear_counterparts_in_this_packet':False,'model_outputs_in_this_packet':False,
           'case_count':32,'sheet_count':8,'sheets':sheets,
           'expected_review_fields':['anonymous_id','decision','reason'],
           'formal_human_FINAL_claim':False,'review_alone_authorizes_training':False}
    dump(PACK/'review_rules.json',rules)
    mapping_path=OUT/'anonymous_input_map_v1.json'
    dump(mapping_path,{'complete':True,'rules':pin(PACK/'review_rules.json'),
                       'shuffle_seed':2026102104,'mapping_only_in_outputs_not_reviewer_packet':True,
                       'cases':mapping,'created_before_any_model_outputs':True})
    manifest={'format':'U96-mild-context-input-preparation-v1','complete':True,
              'created_UTC':datetime.now(timezone.utc).isoformat(),
              'status':'INPUT_PREPARATION_ONLY_PENDING_PROSPECTIVE_INPUT_ONLY_REVIEW',
              'source_script':pin(Path(__file__)),'prior_selection':pin(old_review_path),
              'prior_source_manifest':pin(old_manifest_path),'reference_manifest':pin(original_refs_path),
              'historical_membership_manifest':pin(membership_path),'criteria':pin(CRITERIA),
              'fixed_reference_order':ids,'reference_count':8,'input_case_count':40,
              'degraded_case_count':32,'clear_case_count':8,'role_order':[r[0] for r in ROLE_SIGMA],
              'sources':sources,'cases':cases,
              'operator_specification':{'function':'scipy.ndimage.gaussian_filter','working_dtype':'float64',
                  'mode':'reflect','truncate':3.,'sigma_y_x_channels_by_role':{
                      role:[sigma,sigma,0] if sigma is not None else None for role,sigma in ROLE_SIGMA},
                  'sigmas_used':[1.2,1.8],'quantization':'np.rint ties-to-even, clip[0,255], uint8',
                  'clear':'exact retained256 PNG target bytes',
                  'global':'apply Gaussian to complete immutable clean target once',
                  'local':'replace only[64:192,64:192] with corresponding complete Gaussian image; outside exact',
                  'local_ROI_XYXY_half_open':[64,64,192,192],
                  'corruption_footprint_is_known_not_a_deployment_detector':True,
                  'no_surroundings_preparation_or_protection_selected':True},
              'input_only_packet_rules':pin(PACK/'review_rules.json'),'anonymous_map':pin(mapping_path),
              'input_only_sheets':sheets,
              'overlap':{'historical_DEV_membership_exact':True,'project_TRAIN_source_SHA_overlap':False,
                  'project_TRAIN_photo_URL_overlap':False,'project_TRAIN_ID_overlap':False,
                  'subject_identity_overlap':'unknown; photo/file separation does not prove subject separation',
                  'CodeFormer_pretraining_overlap':'possible/unknown; these FFHQ sources may have been used',
                  'repeated_DEV_exposure':True,'reserved_FINAL_used':False},
              'native_CCTV_used':False,'Zamboanga_performance_claim':False,
              'selected_for_inference':False,'readability_admitted':False,
              'protection_masks_selected':False,'automatic_threshold_calibrated':False,
              'new_source_visual_review_performed':False,'optimization_permitted':False,
              'neural_calls':0,'gradient_calls':0,'optimizer_updates':0,'training_calls':0,
              'VM_calls':0,'HTTP_calls':0,'model_improvement_claimed':False,
              'limits':LIMITS,'versions':{'Python':platform.python_version(),'NumPy':np.__version__,
                  'SciPy':scipy.__version__,'Pillow':PILLOW_VERSION},
              'font':{'absolute_path':str(font_path),'bytes':font_path.stat().st_size,'sha256':sha(font_path.read_bytes())},
              'seconds_to_manifest':time.monotonic()-START}
    manifest_path=OUT/'candidate_manifest_v1.json';dump(manifest_path,manifest)
    # Read all saved members back, including sources, after complete construction.
    for source in sources:
        for key in ['source1024','target256_original','prior_saved_target','target256_saved']:bound(source[key])
    for case in cases:
        image=decode(bound(case['input']),(256,256));target=decode(bound(case['target']),(256,256))
        assert sha(image.tobytes())==case['input_RGB8_sha256']
        if case['role']=='clear':assert bound(case['input'])==bound(case['target'])
        else:
            filtered=blur(target,case['Gaussian_sigma'])
            expected=filtered if case['role'].startswith('global') else target.copy()
            if case['role'].startswith('local'):expected[64:192,64:192]=filtered[64:192,64:192]
            assert np.array_equal(image,expected)
    for item in mapping:assert bound(item['anonymous_input'])==bound(item['original_input'])
    for sheet in sheets:bound(sheet['file'])
    for row in prior_boundaries:bound(row)
    assert 'torch' not in sys.modules and not any(k.startswith('torch.') for k in sys.modules)
    assert len(list(PACK.glob('*.png')))==40 and len(list(OUT.joinpath('inputs').glob('*.png')))==40
    assert not any('target' in p.name or 'map' in p.name for p in PACK.iterdir())
    guard()
    receipt={'complete':True,'saved_input_readback_pass':True,'source_and_target_pins_exact':True,
             'all_prior_boundaries_unchanged':True,'manifest':pin(manifest_path),'script':pin(Path(__file__)),
             'review_rules':pin(PACK/'review_rules.json'),'anonymous_map':pin(mapping_path),
             'reference_count':8,'input_count':40,'degraded_inputs':32,'input_only_sheets':sheets,
             'source1024_to_LANCZOS256_exact':True,'local_outside_ROI_exact':True,
             'clear_bytes_exact':True,'anonymous_bytes_exact':True,
             'readability':'PENDING','selected_for_inference':False,
             'neural_calls':0,'gradient_calls':0,'optimizer_updates':0,'training_calls':0,
             'VM_calls':0,'HTTP_calls':0,'torch_imported':False,
             'generated_bytes_before_receipt':WRITTEN,'elapsed_seconds':time.monotonic()-START,'limits':LIMITS}
    receipt_path=OUT/'preparation_saved_readback_v1.json';dump(receipt_path,receipt)
    print(json.dumps({'complete':True,'script':pin(Path(__file__)),'manifest':pin(manifest_path),
                      'receipt':pin(receipt_path),'inputs':40,'degraded_inputs':32,
                      'sheets':[r['file']['path'] for r in sheets],
                      'generated_bytes':WRITTEN,'seconds':time.monotonic()-START,
                      'neural_training_VM_HTTP_calls':0}))


if __name__=='__main__':main()

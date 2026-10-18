"""CPU-only source selection, exact retained input joins and fixed image features."""
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
import sys
import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION
from . import contract as c


def pixel_sha256(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def input_float32(pixels):
    c.require(pixels.shape==(256,256,3)and pixels.dtype==np.uint8,'Exact RGB8 input required')
    return np.ascontiguousarray(pixels.astype(np.float32)/np.float32(255))


def features(input256,observed):
    c.require(isinstance(input256,np.ndarray)and input256.shape==(256,256,3)and input256.dtype==np.float32 and
        np.isfinite(input256).all()and input256.min()>=0 and input256.max()<=1,'Finite RGB float32[0,1] required')
    c.require(isinstance(observed,np.ndarray)and observed.shape==(256,256)and observed.dtype==np.bool_ and observed.any(),
        'Nonempty boolean support required')
    support=observed.astype(np.float64)
    count=support.reshape(8,32,8,32).sum(axis=(1,3),dtype=np.float64)
    weighted=input256.astype(np.float64)*support[:,:,None]
    total=weighted.reshape(8,32,8,32,3).sum(axis=(1,3),dtype=np.float64)
    means=np.zeros_like(total)
    np.divide(total,count[:,:,None],out=means,where=count[:,:,None]>0)
    planes=np.concatenate((means.transpose(2,0,1),(count/1024)[None]),axis=0)
    return np.ascontiguousarray(planes.reshape(256).astype(np.float32))


def load_rgb(path):
    with Image.open(path)as im:
        c.require(im.mode=='RGB'and im.size==(256,256),'Exact RGB256 PNG required')
        return np.asarray(im).copy()


def load_raw(path):
    with Path(path).open('rb')as stream:
        value=np.load(stream,allow_pickle=False);c.require(not stream.read(1),'Trailing NPY data')
    c.require(value.shape==(256,256,3)and value.dtype==np.float32 and value.flags.c_contiguous and
        np.isfinite(value).all()and value.min()>=0 and value.max()<=1,'Bounded contiguous RAW256 required')
    return value


def reference_index(document):
    c.require(document.get('format')=='codeformer-adaptation-reference-manifest-v1','Unknown approved reference manifest')
    rows=document['references'];c.require(Counter(r['historical_role']for r in rows)=={'train':391,'validation':53},'Approved split census differs')
    result={r['id']:r for r in rows};c.require(len(result)==444,'Duplicate photographic reference')
    for row in rows:
        c.require(row['decision']=='accept_clean_restoration'and row['native_size']==[1024,1024]and
            c.HEX.fullmatch(row['source_sha256']),'Unapproved photographic source')
        c.safe_relative(row['source_path'])
        if row['historical_role']=='train':c.require(row['suitable_as_hq_train_candidate']is True,'Unapproved TRAIN source')
    a={r['source_sha256']for r in rows if r['historical_role']=='train'}
    b={r['source_sha256']for r in rows if r['historical_role']=='validation'}
    c.require(not a&b,'Exact TRAIN/DEV source overlap')
    return result


def select_heldout(references,parity,exclusions):
    c.require(parity.get('TRAIN_only')is True and len(parity['cases'])==200 and
        [(r['update'],r['slot'])for r in parity['cases']]==[(u,s)for u in range(1,51)for s in range(1,5)],
        'Exact200 chronological fitted observations required')
    fitted_ids={r['reference_id']for r in parity['cases']};fitted_hashes={r['source_sha256']for r in parity['cases']}
    c.require(len(fitted_ids)==100 and len(fitted_hashes)==100,'Exact100 fitted photographic sources required')
    excluded_ids=set(exclusions['reference_ids'])|fitted_ids
    excluded_hashes=set(exclusions['source_sha256'])|fitted_hashes
    for rid in fitted_ids:c.require(references[rid]['historical_role']=='train','Fitted DEV/FINAL source')
    candidates=[r for r in references.values()if r['historical_role']=='train'and r['id']not in excluded_ids and r['source_sha256']not in excluded_hashes]
    candidates=sorted(candidates,key=lambda r:(r['source_sha256'],r['id']))
    c.require(len(candidates)>=32 and len({r['source_sha256']for r in candidates})==len(candidates),'Insufficient distinct unused TRAIN sources')
    selected=[]
    for index,row in enumerate(candidates[:32]):
        selected.append({'reference_id':row['id'],'source_path':row['source_path'],'source_sha256':row['source_sha256'],
            'camera_update':1+3*(index//2),'pair_role':'A'if index%2==0 else'B','pair_index':index//2,
            'canonical_rank':index,'reference':row})
    return selected


def render_heldout(selected,reference_sha,resolver,historical_pins):
    """Exact approved CPU generator; no sampler, model or framework import."""
    for name in (*c.CAMERA_FILES,'research/codeformer_adaptation_v1/__init__.py','research/codeformer_adaptation_v2/__init__.py'):
        c.require(c.sha256(Path(__file__).resolve().parents[2]/name)==historical_pins[name]['sha256'],
            'Current camera source differs from pinned historical bytes')
    from research.codeformer_adaptation_v2 import data as camera
    import cv2
    c.require(not any(n in sys.modules for n in ('torch','tensorflow','jax')),'Neural framework imported by camera path')
    payloads={};cases=[]
    for offset in range(0,32,2):
        entries=selected[offset:offset+2];refs=[]
        for entry in entries:
            row=entry['reference'];path=Path(resolver('adain',row['source_path'])).resolve(strict=True)
            c.require(c.sha256(path)==row['source_sha256'],'Selected source bytes changed')
            relative=Path(row['source_path']);workspace=path
            for _ in relative.parts:workspace=workspace.parent
            c.require(workspace/relative==path,'Physical source overlay does not retain declared suffix')
            refs.append(camera.Reference(id=row['id'],role='train',workspace=workspace,
                source_relative_path=row['source_path'],source_sha256=row['source_sha256'],
                clean512_rgb8_sha256=row['proposed_512_rgb_sha256'],manifest_sha256=reference_sha,
                attribution_json=json.dumps(row['attribution'],sort_keys=True,separators=(',',':'))))
        generated=camera.make_training_update(refs[0],refs[1],entries[0]['camera_update'])
        for pair in generated:
            record=pair.record;rid=record['reference_id'];role=record['member'];stem='inputs/heldout_'+rid+'/'+role+'/'
            pixels=np.rint(pair.input256*np.float32(255)).astype(np.uint8)
            c.require(pixel_sha256(pixels)==record['input256_rgb8_sha256']and
                pixel_sha256(input_float32(pixels))==record['input256_float32_sha256'],'Generated RGB8/255 parity differs')
            stream=BytesIO();Image.fromarray(pixels).save(stream,format='PNG');raw=stream.getvalue()
            c.require(len(raw)<=c.LIMITS['one256PNG_bytes'],'Generated PNG exceeds declared cap')
            pair_raw=(json.dumps(record,indent=2,allow_nan=False)+'\n').encode('utf-8')
            payloads[stem+'input.png']=raw;payloads[stem+'pair_record.json']=pair_raw
            cases.append({'id':'heldout_'+rid+'_'+role,'cohort':'heldout','reference_id':rid,'role':role,
                'pair_id':record['pair_id'],'camera_update':record['update'],'pair_role':record['pair_role'],'slot':record['slot'],
                'input_path':stem+'input.png','input_sha256':hashlib.sha256(raw).hexdigest(),
                'pair_record_path':stem+'pair_record.json','pair_record_sha256':hashlib.sha256(pair_raw).hexdigest(),
                'source_path':record['source_path'],'source_sha256':record['source_sha256'],
                'input256_rgb8_sha256':record['input256_rgb8_sha256'],'input256_float32_sha256':record['input256_float32_sha256']})
    environment={'Python':sys.version,'platform':platform.platform(),'NumPy':np.__version__,'Pillow':PILLOW_VERSION,'OpenCV':cv2.__version__,
        'neural_framework_imported':False,'renderer_files_sha256':{n:historical_pins[n]['sha256']for n in c.CAMERA_FILES},
        'source_photographs_decoded':32,'DEV_FINAL_native_pixels_opened':0}
    return {'format':'codeformer-fixed-state-heldout-pairs-v1','cases':cases,'render_environment':environment},payloads


def load_index(root,protocol,parents,*,resolver=None):
    c.verify_protocol(protocol);root=Path(root);checked={}
    def bound(alias,name):
        pin=protocol['dependencies'][alias][name]
        path=c.safe_path(parents[alias],name)if resolver is None else c.no_links(resolver(alias,name))
        key=alias+':'+name
        if key not in checked:
            c.require(path.stat().st_size==pin['bytes']and c.sha256(path)==pin['sha256'],'Exact input dependency differs: '+key)
            checked[key]=pin['sha256']
        return path
    def own(name):
        path=c.safe_path(root,name)
        c.require(path.stat().st_size==protocol['files_bytes'][name]and c.sha256(path)==protocol['files_sha256'][name],
            'Exact own data artifact differs')
        return path
    references=reference_index(c.read_json(bound('adain','references.json')))
    parity=c.read_json(bound('adain','training_parity_v1.json'))
    fitted_doc=c.read_json(own(protocol['fitted_index_file']));held_doc=c.read_json(own(protocol['new_pairs_file']))
    selection=c.read_json(own(protocol['selection_file']));exclusions=c.read_json(own(protocol['exclusions_file']))
    selected=select_heldout(references,parity,exclusions)
    c.require(c.equal(selection['selected'],selected),'Canonical heldout selection differs')
    c.require([r['id']for r in held_doc['cases']]==['heldout_'+r['reference_id']+'_'+role for r in selected for role in ('normal','dark')],
        'Canonical heldout normal/dark order differs')
    c.require(held_doc['selection_sha256']==protocol['files_sha256'][protocol['selection_file']]and
        set(c.expected_pair_paths(held_doc))<=set(protocol['files_sha256']),'Heldout source/input closure differs')
    c.expected_row_ids(fitted_doc,held_doc)
    fitted=[];heldout=[];w=np.asarray([.2126,.7152,.0722],np.float64)
    for row,p in zip(fitted_doc['cases'],parity['cases']):
        ref=references[p['reference_id']]
        record=c.read_json(bound('adain',p['pair_record_path']))
        cache=c.read_json(bound('tone_training',row['cache_path']))
        role='normal'if p['slot']in (1,3)else'dark'
        c.require(row['id']=='train_u%04d_slot%d'%(p['update'],p['slot'])and row['role']==role and
            row['reference_id']==p['reference_id']and row['source_sha256']==p['source_sha256']and
            ref['historical_role']=='train'and ref['source_sha256']==p['source_sha256']and
            ref['source_path']==p['clean_source_path']==record['source_path']and
            record['reference_manifest_sha256']==protocol['dependencies']['adain']['references.json']['sha256']and
            record['reference_id']==p['reference_id']and record['source_sha256']==p['source_sha256']and
            record['member']==role and record['update']==p['update']and record['slot']==p['slot']and
            record['input256_rgb8_sha256']==p['input256_rgb8_sha256']and
            all(cache['sourcebindings'][k]==p[k]for k in cache['sourcebindings']), 'Fitted data/source/role join differs')
        pixels=load_rgb(bound('tone_training',row['input_path']));floats=input_float32(pixels)
        c.require(pixel_sha256(pixels)==p['input256_rgb8_sha256']and pixel_sha256(floats)==p['input256_float32_sha256'],
            'Fitted retained RGB8 pixels differ')
        base=load_raw(bound('tone_training',row['base_raw_path']));original=load_raw(bound('tone_training',row['original_raw_path']))
        c.require(cache['id']==row['id']and cache['role']==role and cache['update']==row['update']==p['update']and
            cache['slot']==row['slot']==p['slot']and cache['target_source']==('original'if role=='normal'else'full_coarse_base')and
            cache['base_raw']['path']==row['base_raw_path']and cache['original_raw']['path']==row['original_raw_path'],
            'Fitted target/parent cache join differs')
        for key in ('base_raw','original_raw'):
            pin=protocol['dependencies']['tone_training'][cache[key]['path']]
            c.require({k:cache[key][k]for k in ('sha256','bytes')}==pin,'Fitted parent RAW evidence binding differs')
        target=cache['target_mean_y'];mean=float(((original if role=='normal'else base).astype(np.float64)@w).mean())
        c.require(abs(target-mean)<=2e-7,'Fitted cached target is not declared parent mean')
        observed=np.ones((256,256),bool);feature=features(floats,observed)
        c.require(np.array_equal(feature,np.asarray(cache['features'],np.float32)),'Fitted shared input features differ')
        fitted.append(dict(row,input_rgb8=pixels,input256=floats,observed=observed,features=feature,source='FFHQ fitted TRAIN',
            evidence='paired_synthetic_TRAIN_response',role_metadata='Audit target selection only; never a controller inference input',
            base_raw=base,original_raw=original,target=target,pair_id=record['pair_id'],profile=record['profile'],sourcebindings=p))
    selected_map={r['reference_id']:r for r in selected}
    for row in held_doc['cases']:
        record=c.read_json(own(row['pair_record_path']));entry=selected_map[row['reference_id']]
        c.require(record['reference_id']==row['reference_id']and record['source_sha256']==row['source_sha256']==entry['source_sha256']and
            record['source_path']==entry['source_path']and record['member']==row['role']and
            record['update']==row['camera_update']==entry['camera_update']and record['pair_role']==entry['pair_role']and
            record['pair_id']==row['pair_id']and record['slot']==row['slot']and record['role']=='train'and
            record['reference_manifest_sha256']==protocol['dependencies']['adain']['references.json']['sha256'],
            'Heldout source/role/camera/record join differs')
        pixels=load_rgb(own(row['input_path']));floats=input_float32(pixels)
        c.require(pixel_sha256(pixels)==row['input256_rgb8_sha256']==record['input256_rgb8_sha256']and
            pixel_sha256(floats)==row['input256_float32_sha256']==record['input256_float32_sha256']and
            protocol['files_sha256'][row['input_path']]==row['input_sha256']and
            protocol['files_sha256'][row['pair_record_path']]==row['pair_record_sha256'],'Heldout exact pixels/record hash differs')
        observed=np.ones((256,256),bool)
        heldout.append(dict(row,input_rgb8=pixels,input256=floats,observed=observed,features=features(floats,observed),source='FFHQ unused TRAIN',
            evidence='paired_synthetic_TRAIN_response',role_metadata='Audit target selection only; never a controller inference input',
            original_raw=None,base_raw=None,target=None,profile=record['profile'],sourcebindings=record))
    c.require(len(fitted)==200 and len(heldout)==64 and Counter(r['role']for r in heldout)=={'normal':32,'dark':32},'Full finite index census differs')
    return {'fitted':fitted,'heldout':heldout,'binding_receipt':{'complete':True,'fitted_observations':200,
        'fitted_reference_sources':100,'heldout_observations':64,'heldout_reference_sources':32,
        'fitted_parent_RAW_arrays':400,'checked_files_sha256':checked,'all_observed_masks_full_true':True,
        'reference_ids_are_not_proven_person_ids':True,'new_DEV_FINAL_pixels_opened':0,'neural_calls':0}}


def build_index(protocol,parents,*,root=None,resolver=None):
    return load_index(Path(__file__).resolve().parents[2]if root is None else root,protocol,parents,resolver=resolver)

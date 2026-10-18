"""Independent saved-pixel metrics for a fixed filter comparison; no neural runtime."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import stat
import sys

import numpy as np
from PIL import Image, ImageDraw, __version__ as PILLOW_VERSION
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
DATA_SHA = '7823847ce7a709308fe1f4d35d35e2c14f752b2d5c303f5fa1cb4b7c8de8c3a8'
REVIEW_SHA = '51bff1593cd33ad2b369a7c20d2b5282ed67521df4b62e17a53f082941fc3b50'
Y = np.asarray([.2126,.7152,.0722], np.float64)
ARMS = ('original0','parent50','candidate50')
FLAGGED_REFERENCE = 'va_ffhq_08025'


def require(value, reason):
    if not value:
        raise ValueError(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def unique_json(data):
    def unique(items):
        result = {}
        for key,value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(data.decode('utf-8'), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def normalized_rgb(array):
    require(isinstance(array,np.ndarray) and array.ndim==3 and array.shape[2]==3 and
            min(array.shape[:2])>0 and array.dtype in (np.dtype('float32'),np.dtype('float64')) and
            np.isfinite(array).all() and array.min()>=0 and array.max()<=1,
            'Finite normalized RGB floats required')
    return array.astype(np.float64)


def residual_metrics(predicted, clean, support):
    """Reference disagreement; offset/gradient diagnostics are not acceptance gates."""
    predicted,clean=normalized_rgb(predicted),normalized_rgb(clean)
    require(predicted.shape==clean.shape and isinstance(support,np.ndarray) and
            support.dtype==np.bool_ and support.shape==clean.shape[:2] and support.any(),
            'Aligned clean geometry and nonempty boolean observed support required')
    residual=predicted-clean
    observed=residual[support]
    offset=observed.mean(axis=0,dtype=np.float64)
    centered=observed-offset
    mae=float(np.abs(observed).mean(dtype=np.float64))
    mse=float(np.square(observed).mean(dtype=np.float64))
    centered_mse=float(np.square(centered).mean(dtype=np.float64))
    energy=float(np.square(offset).mean(dtype=np.float64))
    closure=mse-(centered_mse+energy)
    valid_x=support[:,1:] & support[:,:-1]
    valid_y=support[1:,:] & support[:-1,:]
    nx,ny=int(valid_x.sum()),int(valid_y.sum())
    dx=np.diff(residual,axis=1)[valid_x]
    dy=np.diff(residual,axis=0)[valid_y]
    sum_x=float(np.abs(dx).sum(dtype=np.float64))
    sum_y=float(np.abs(dy).sum(dtype=np.float64))
    if support.all() and min(clean.shape[:2])>=7:
        ssim=float(structural_similarity(predicted,clean,data_range=1.,channel_axis=2,win_size=7,
                                        gaussian_weights=False,use_sample_covariance=True))
    else:
        ssim=None
    return {'mae':mae,'mse':mse,'psnr':None if mse==0 else -10*math.log10(mse),
        'PSNR_zero_MSE_policy':'null denotes mathematically infinite PSNR; no finite fabricated value',
        'ssim':ssim,'SSIM_policy':'skimage structural_similarity uniform7x7,sample covariance,data_range1,RGB channel mean; only all-observed case',
        'signed_luminance_bias':float((observed@Y).mean(dtype=np.float64)),
        'absolute_luminance_bias':float(abs((observed@Y).mean(dtype=np.float64))),
        'mean_RGB_offset':offset.tolist(),'mean_centered_residual_mae':float(np.abs(centered).mean(dtype=np.float64)),
        'mean_centered_residual_mse':centered_mse,'global_RGB_offset_energy':energy,
        'global_RGB_offset_energy_share':None if mse==0 else energy/mse,
        'MSE_decomposition_closure_residual':closure,
        'adjacent_RGB_residual_gradient_mae':None if nx+ny==0 else (sum_x+sum_y)/(3*(nx+ny)),
        'adjacent_RGB_residual_gradient_x_mae':None if nx==0 else sum_x/(3*nx),
        'adjacent_RGB_residual_gradient_y_mae':None if ny==0 else sum_y/(3*ny),
        'adjacent_x_pairs':nx,'adjacent_y_pairs':ny,'observed_pixels':int(support.sum()),
        'offset_diagnostics_are_not_acceptance_or_identity_metrics':True}


def file_bytes(path, expected_sha=None, expected_size=None, cap=8*1024**2):
    path=Path(path).absolute()
    require(path.resolve(strict=True).is_relative_to(ROOT),'Evidence outside workspace')
    for part in (path,*path.parents):
        if part==ROOT.parent:break
        info=part.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info,'st_file_attributes',0)&1024,
                'No linked/reparse evidence')
    before=path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size<=cap and
            (expected_size is None or before.st_size==expected_size),'Ordinary bounded exact file required')
    data=path.read_bytes();after=path.stat()
    require((before.st_ino,before.st_mtime_ns,before.st_size)==(after.st_ino,after.st_mtime_ns,after.st_size),
            'Evidence changed across read')
    require(expected_sha is None or digest(data)==expected_sha,'Evidence SHA differs')
    return data


def canonical_relative(name):
    require(isinstance(name,str) and name and ':' not in name and '\\' not in name,'Canonical relative path')
    p=PurePosixPath(name)
    require(not p.is_absolute() and p.as_posix()==name and all(x not in ('','.','..') for x in p.parts),
            'Canonical relative path')
    return name


def rgb_png(data):
    with Image.open(io.BytesIO(data)) as image:
        require(image.format=='PNG' and image.mode=='RGB' and image.size==(256,256) and
                getattr(image,'n_frames',1)==1,'Exact single-frame RGB256 PNG required')
        return np.asarray(image).copy()


def raw_rgb(data):
    require(len(data)<=3*256*256*4+256,'Bounded RAW required')
    stream=io.BytesIO(data); value=np.load(stream,allow_pickle=False)
    require(not stream.read(1),'Trailing RAW bytes')
    require(value.dtype==np.float32 and value.shape==(256,256,3) and value.flags.c_contiguous,
            'Exact contiguous FP32 RGB256 RAW required')
    normalized_rgb(value)
    return value


def scalar_mean_metrics(rows):
    keys=('mae','mse','psnr','ssim','signed_luminance_bias','absolute_luminance_bias',
          'mean_centered_residual_mae','mean_centered_residual_mse','global_RGB_offset_energy',
          'global_RGB_offset_energy_share','adjacent_RGB_residual_gradient_mae')
    return {key:{'mean':None if not any(r[key] is not None for r in rows) else
                float(np.mean([r[key] for r in rows if r[key] is not None],dtype=np.float64)),
                'finite_cases':sum(r[key] is not None for r in rows),'null_cases':sum(r[key] is None for r in rows)}
            for key in keys}


def source_stratum(reference_id):
    return 'appearance_flag_va_ffhq_08025' if reference_id==FLAGGED_REFERENCE else 'other_seven_photographs'


# Saved-output worker schema is reviewed independently before this module gains
# its bounded actual-artifact audit entry point. No inference entry point exists.


SETTINGS = ('w0_adain0','w0_adain1','w1_adain0','w1_adain1')
CASE_IDS = ('tr_ffhq_00084__clear','tr_ffhq_00084__repeated_rows_4_blend0p75',
    'va_ffhq_00383__local_gaussian_3p0','va_ffhq_00383__local_mosaic_8','va_ffhq_09056__global_mosaic_8')
LOCAL_CASES = frozenset(CASE_IDS[2:4])
CONTRASTS = {
    'AdaIN_effect_at_w0': {'w0_adain1':1.,'w0_adain0':-1.},
    'AdaIN_effect_at_w1': {'w1_adain1':1.,'w1_adain0':-1.},
    'fusion_effect_at_AdaINoff': {'w1_adain0':1.,'w0_adain0':-1.},
    'fusion_effect_at_AdaINon': {'w1_adain1':1.,'w0_adain1':-1.},
    'interaction_Y11_Y10_Y01_Y00': {'w1_adain1':1.,'w1_adain0':-1.,'w0_adain1':-1.,'w0_adain0':1.},
}
SCALAR_METRICS = ('mae','mse','psnr','ssim','signed_luminance_bias','absolute_luminance_bias',
    'mean_centered_residual_mae','mean_centered_residual_mse','global_RGB_offset_energy',
    'global_RGB_offset_energy_share','adjacent_RGB_residual_gradient_mae')


def contrast_maps(outputs):
    require(set(outputs)==set(SETTINGS),'Exact four factorial conditions required')
    arrays={name:normalized_rgb(value) for name,value in outputs.items()}
    require(len({a.shape for a in arrays.values()})==1,'Same fixed image geometry required')
    return {name:sum(weight*arrays[condition] for condition,weight in weights.items())
            for name,weights in CONTRASTS.items()}


def signed_map_statistics(array,support):
    """Signed setting interactions are not normalized images and must not be clipped."""
    require(isinstance(array,np.ndarray) and array.ndim==3 and array.shape[2]==3 and
            array.dtype==np.float64 and np.isfinite(array).all() and support.dtype==np.bool_ and
            support.shape==array.shape[:2] and support.any(),'Finite signed map and observed support required')
    values=array[support]
    valid_x=support[:,:-1]&support[:,1:];valid_y=support[:-1,:]&support[1:,:]
    pairs=int(valid_x.sum()+valid_y.sum())
    total=float(np.abs(np.diff(array,axis=1)[valid_x]).sum(dtype=np.float64)+
                np.abs(np.diff(array,axis=0)[valid_y]).sum(dtype=np.float64))
    return {'minimum':float(values.min()),'maximum':float(values.max()),
        'mean_RGB':values.mean(axis=0,dtype=np.float64).tolist(),
        'signed_mean_luminance':float((values@Y).mean(dtype=np.float64)),
        'mean_absolute_RGB':float(np.abs(values).mean(dtype=np.float64)),
        'maximum_absolute_RGB':float(np.abs(values).max()),
        'RMS_RGB':float(np.sqrt(np.square(values).mean(dtype=np.float64))),
        'nonzero_channel_values':int(np.count_nonzero(values)),'observed_pixels':int(support.sum()),
        'adjacent_x_pairs':int(valid_x.sum()),'adjacent_y_pairs':int(valid_y.sum()),
        'adjacent_signed_map_gradient_mae':None if pairs==0 else total/(3*pairs),
        'clipping_applied':False,'normalized_restoration_image':False}


def region_masks(case_id,footprint):
    require(case_id in CASE_IDS and isinstance(footprint,np.ndarray) and footprint.shape==(256,256)
            and footprint.dtype==np.uint8 and set(np.unique(footprint)).issubset({0,255}),
            'Declared case and binary source-bound footprint required')
    regions={'full_observed':np.ones((256,256),bool)}
    if case_id in LOCAL_CASES:
        expected=np.zeros((256,256),np.uint8);expected[64:192,64:192]=255
        require(np.array_equal(footprint,expected),'Exact local corruption footprint required')
        regions['inside_corruption_footprint']=footprint>0
        regions['outside_corruption_footprint']=footprint==0
    return regions


def factorial_diagnostics(outputs,clean,regions):
    maps=contrast_maps(outputs)
    answer={}
    for region,support in regions.items():
        metrics={name:residual_metrics(value,clean,support) for name,value in outputs.items()}
        contrasts={}
        for name,weights in CONTRASTS.items():
            metric_differences={key:None if any(metrics[setting][key] is None for setting in weights)
                else float(sum(weight*metrics[setting][key] for setting,weight in weights.items()))
                for key in SCALAR_METRICS}
            contrasts[name]={'signed_pixel_map':signed_map_statistics(maps[name],support),
                'metric_contrast_not_metric_of_interaction_map':metric_differences}
        answer[region]={'support_pixels':int(support.sum()),'paired_setting_metrics':metrics,'setting_contrasts':contrasts,
            'region_is_anatomical_or_identity_mask':False,'acceptance_gate_or_optimal_setting_selected':False}
    return answer


# Actual saved-artifact schema is appended only after independently reviewed
# worker/protocol definitions. No model or inference entry point is provided.

METRIC_CONTRACT_SHA = '5462dc6810b8183fed7d3ee2ba32c15520edc4973e7dbb3e91f3b989de27f657'
HELPER_PREFIX_SHA = 'be6ebb9a713610a089bea5209f57883d72df73445f8e3eb277d565a547e23b74'
OLD_HELPER_SHA = '75f96ffe5e7438beca734119c5e763b4ae2fe39ffc6181afc194a5e274252506'
MODEL_BINDING_SHA = '5a926fc23c58916374d4ad69bf68fc4066bc469c440d920a2046b09a2761a994'
REVIEWED_CAPS = {'new_forward_calls':48,'wall_seconds':600,'process_RSS_bytes':3*1024**3,
    'minimum_available_RAM_bytes':768*1024**2,'minimum_disk_free_bytes':1024**3,
    'output_payload_bytes':64*1024**2,'output_files':170,'journal_records':180,
    'journal_bytes':1024**2,'watchdog_period_seconds':1}
FUSIONS = ('32','64','128','256')
SETTING_VALUES = {name:{'id':name,'w':float(name[1]),'adain':name[-1]=='1'} for name in SETTINGS}
MODELS = {
    'original0': ('original','1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7',
        '6cb93cee937f67814edcf4625a68bae4bf8f1c2351f7306864ceefa63d4803bd'),
    'parent50': ('parent','53778f54589697a14b067033eff8cffc2985985e8a33021092de3c14fbf443bc',
        '92ac81750b86ef1da6f1f7c0480b9e64773fafe07b094e4a5ad7e88b3f7a0c8b'),
    'candidate50': ('r2','d4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896',
        'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f')}


def sha_string(value):
    return isinstance(value,str) and len(value)==64 and set(value)<=set('0123456789abcdef')


def check_floor(raw,png):
    require(np.array_equal(png,np.floor(raw*np.float32(255)).astype(np.uint8)),
        'Saved PNG differs from exact FP32 RAW floor conversion')


def observed_setting(observer,setting):
    require(setting in SETTING_VALUES and isinstance(observer,dict),'Known setting and observer required')
    expected=SETTING_VALUES[setting]
    require(set(observer)=={'top_level_calls','top_level_settings','fusion_calls','fusion_order'} and
        type(observer['top_level_calls']) is int and observer['top_level_calls']==1 and
        observer['top_level_settings']==[{'w':expected['w'],'adain':expected['adain'],'detach_16':True}],
        'One observed top-level call with exact late-path settings required')
    expected_count=1 if expected['w']==1 else 0
    require(set(observer['fusion_calls'])==set(FUSIONS) and
        all(type(x) is int and x==expected_count for x in observer['fusion_calls'].values()) and
        observer['fusion_order']==(list(FUSIONS) if expected_count else []),
        'Observed fusion bypass/enable order differs')


def readout_signature(readout):
    require(isinstance(readout,dict) and set(readout)=={'logits','lq_feature','recomputed_tokens',
        'top2_probability_margin','readout_only_no_extra_neural_forward'},'Exact returned readout schema required')
    for key,shape in [('logits',[1,256,1024]),('lq_feature',[1,256,16,16])]:
        value=readout[key]
        require(set(value)=={'shape','dtype','array_sha256'} and value['shape']==shape and
            value['dtype']=='torch.float32' and sha_string(value['array_sha256']),
            'Returned internal shape/dtype/hash differs')
    tokens=readout['recomputed_tokens']
    require(set(tokens)=={'shape','dtype','array_sha256','indices','rule','topk_native_shape',
        'actual_internal_codebook_lookup_captured'} and tokens['shape']==[1,256] and tokens['dtype']=='int64' and
        tokens['topk_native_shape']==[1,256,1] and tokens['actual_internal_codebook_lookup_captured'] is False and
        tokens['rule']=='Exact released F.softmax(returned_logits,dim2) then torch.topk(k1,dim2); squeeze trailing singleton',
        'Recomputed token rule/shape/capture claim differs')
    require(isinstance(tokens['indices'],list) and len(tokens['indices'])==256 and
        all(type(v) is int and 0<=v<=1023 for v in tokens['indices']) and
        digest(np.asarray(tokens['indices'],dtype=np.int64).reshape(1,256).tobytes())==tokens['array_sha256'],
        'Exact256 bounded int64 recomputed token fingerprint differs')
    margin=readout['top2_probability_margin']
    require(set(margin)=={'minimum','mean','maximum','exact_zero_margin_token_count'} and
        all(type(margin[k]) in (float,int) and math.isfinite(margin[k]) for k in ('minimum','mean','maximum')) and
        0<=margin['minimum']<=margin['maximum']<=1 and 0<=margin['mean']<=1 and
        type(margin['exact_zero_margin_token_count']) is int and 0<=margin['exact_zero_margin_token_count']<=256 and
        (margin['minimum']==0)==(margin['exact_zero_margin_token_count']>0) and
        (margin['maximum']==0)==(margin['exact_zero_margin_token_count']==256) and
        readout['readout_only_no_extra_neural_forward'] is True,'Finite source-bound top2 margin summary differs')
    return {key:readout[key] for key in ('logits','lq_feature','recomputed_tokens')}


def validate_resource(sample):
    require(set(sample)=={'process_RSS_bytes','available_RAM_bytes','disk_free_bytes'} and
        all(type(v) is int and v>0 for v in sample.values()) and
        sample['process_RSS_bytes']<=REVIEWED_CAPS['process_RSS_bytes'] and
        sample['available_RAM_bytes']>=REVIEWED_CAPS['minimum_available_RAM_bytes'] and
        sample['disk_free_bytes']>=REVIEWED_CAPS['minimum_disk_free_bytes'],
        'Recorded resource sample violates prospective finite envelope')


def expected_calls():
    rows=[]
    for arm in ARMS:
        rows.append((arm,CASE_IDS[0],'w1_adain1','fresh_clear_parity'))
        rows.extend((arm,key,setting,'new_factorial') for key in CASE_IDS for setting in SETTINGS[:3])
    return rows


def validate_journal(journal,evidence_by_call,proofs):
    require(len(journal)==154 and len(evidence_by_call)==48,'Exact154 durable journal events and48 evidence calls required')
    cursor=0;call=0;resources=[]
    def pop(event):
        nonlocal cursor
        require(cursor<len(journal) and journal[cursor]['event']==event,'Ordered durable journal event differs')
        value=journal[cursor];cursor+=1
        if 'resources' in value:
            validate_resource(value['resources']);resources.append(value['resources'])
        return value
    for arm in ARMS:
        loaded=pop('model_loaded_frozen')
        require(loaded['arm']==arm and loaded['state_sha256']==MODELS[arm][2],'Loaded state journal binding differs')
        plan=[(CASE_IDS[0],'w1_adain1','fresh_clear_parity')]+[
            (key,setting,'new_factorial') for key in CASE_IDS for setting in SETTINGS[:3]]
        for local_call,(key,setting,kind) in enumerate(plan):
            call+=1;reserve=pop('forward_reserved');returned=pop('forward_returned');commit=pop('output_committed');e=evidence_by_call[call]
            require(reserve['call']==returned['call']==commit['call']==e['call']==call and
                all(r['arm']==arm and r['id']==key and r['kind']==kind for r in (reserve,returned,commit,e)) and
                reserve['setting']==e['setting']==SETTING_VALUES[setting] and
                returned['setting']==commit['setting']==setting and
                returned['observer']==commit['observer']==e['observer'] and commit['evidence']==e['_pin'],
                'Reserve-return-commit exact call/evidence join differs')
            require(type(reserve['seconds']) in (float,int) and math.isfinite(reserve['seconds']) and
                0<=reserve['seconds']<=600,'Reservation timing outside finite envelope')
            observed_setting(returned['observer'],setting)
            if local_call==0:
                passed=pop('baseline_parity_passed')
                require(passed['arm']==arm and passed['id']==CASE_IDS[0] and passed['before_interventions'] is True,
                    'Clear parity must pass before model interventions')
        finished=pop('model_finished_frozen')
        require(finished['arm']==arm and finished['proof']==proofs[arm],'Finished model proof differs')
    end=pop('complete');progress=end['progress']
    require(cursor==len(journal) and progress=={'reserved':48,'returned':48,'committed':48,'in_flight':None,'arm':'r2'},
        'Completed journal does not close exact48 calls')
    return resources


def workspace_pin(pin,load=True,original=False):
    canonical_relative(pin['path'])
    path=(ROOT/pin['path']).absolute()
    require(path==Path(pin['absolute_path']).absolute(),'Physical workspace pin differs')
    if pin['bytes']<=8*1024**2:
        value=file_bytes(path,pin['sha256'],pin['bytes']);return value if load else None
    require(not load and pin['bytes']<512*1024**2,'Only inert bounded model streaming allowed for large evidence')
    for part in (path,*path.parents):
        if part==ROOT.parent:break
        info=part.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info,'st_file_attributes',0)&1024,'Linked model path')
    before=path.stat();allowed=3 if original else 1
    require(stat.S_ISREG(before.st_mode) and before.st_nlink==allowed and before.st_size==pin['bytes'],
        'Exact model bytes/link count required')
    if original:
        require(pin['path']=='outputs/codeformer_restoration_pretrained_v1/codeformer.pth' and pin['sha256']==MODELS['original0'][1],
            'Only exact untouched benchmark permits three hardlinks')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    after=path.stat()
    require(h.hexdigest()==pin['sha256'] and
        (before.st_size,before.st_ino,before.st_mtime_ns,before.st_nlink)==
        (after.st_size,after.st_ino,after.st_mtime_ns,after.st_nlink),'Streamed model bytes changed')
    return None


def footprint_png(payload):
    with Image.open(io.BytesIO(payload)) as image:
        require(image.format=='PNG' and image.mode=='L' and image.size==(256,256) and
            getattr(image,'n_frames',1)==1,'Exact binary256 corruption footprint required')
        values=np.asarray(image).copy()
    require(set(np.unique(values)).issubset({0,255}),'Corruption footprint is not binary')
    return values


def save_json_exclusive(path,value):
    payload=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode('utf-8')
    with path.open('xb') as stream:stream.write(payload)
    require(path.read_bytes()==payload,'Audit output readback differs')
    return {'path':str(path),'sha256':digest(payload),'bytes':len(payload)}


def render_sheets(out,cases,inputs,targets,png_bank):
    destination=out/'sheets';destination.mkdir();pages=[];cells=[]
    titles=('Observed / resize control','Known paired synthetic target','w0 / AdaIN off','w0 / AdaIN on',
        'w1 / AdaIN off','w1 / AdaIN on - cached')
    for number,case in enumerate(cases,1):
        key=case['id'];sheet=Image.new('RGB',(1536,916),'white');draw=ImageDraw.Draw(sheet)
        draw.text((4,2),key+' | exploratory fixed-path comparison | exact saved256 pixels | no new model qualification',fill='black')
        for col,title in enumerate(titles):draw.text((col*256+3,23),title,fill='black')
        page_cells=[]
        for row,arm in enumerate(ARMS):
            y=52+row*288
            draw.text((4,y),arm+' | '+case['split']+' | input-only '+case['input_review']['decision']+' | no enhancement or resampling',fill='black')
            tiles=[inputs[key],targets[key]]+[png_bank[arm,key,setting] for setting in SETTINGS]
            for col,tile in enumerate(tiles):
                require(tile.dtype==np.uint8 and tile.shape==(256,256,3),'Exact256 PNG sheet panel required')
                x,py=col*256,y+24;sheet.paste(Image.fromarray(tile),(x,py))
                page_cells.append({'id':key,'arm':arm,'column':titles[col],'setting':None if col<2 else SETTINGS[col-2],
                    'x':x,'y':py,'width':256,'height':256,'pixel_sha256':digest(tile.tobytes())})
        stream=io.BytesIO();sheet.save(stream,format='PNG');payload=stream.getvalue()
        path=destination/('%02d_%s.png'%(number,key))
        with path.open('xb') as saved:saved.write(payload)
        with Image.open(io.BytesIO(path.read_bytes())) as saved:
            pixels=np.asarray(saved)
            for cell in page_cells:
                require(digest(pixels[cell['y']:cell['y']+256,cell['x']:cell['x']+256].tobytes())==cell['pixel_sha256'],
                    'Encoded comparison sheet does not preserve exact PNG panel')
        pin={'path':path.relative_to(out).as_posix(),'sha256':digest(payload),'bytes':len(payload),
            'width':1536,'height':916,'model_rows':3,'columns':6,'id':key}
        pages.append(pin);cells.extend(dict(cell,sheet=pin['path']) for cell in page_cells)
    require(len(pages)==5 and len(cells)==90,'Exact five sheets/ninety saved256 panels required')
    return {'complete':True,'sheets':pages,'cells':cells,'exact256_panels':90,'factorial_model_panels':60,
        'repeated_input_and_target_panels':30,'fresh_parity_panels_not_duplicated':True,
        'RAW_metrics_not_visually_displayed':True,'root_all5_visual_review':'PENDING',
        'no_display_enhancement_or_resampling':True,'model_qualified':False}


def aggregate_conditional_effects(groups):
    rows=[]
    for arm in ARMS:
        selected=[g for g in groups if g['arm']==arm]
        require(len(selected)==5,'All five purposive cases per arm required')
        for view in ('raw','png'):
            for contrast in CONTRASTS:
                metrics={}
                for key in SCALAR_METRICS:
                    values=[g['views'][view]['full_observed']['setting_contrasts'][contrast][
                        'metric_contrast_not_metric_of_interaction_map'][key] for g in selected]
                    finite=[v for v in values if v is not None]
                    metrics[key]={'mean':None if not finite else float(np.mean(finite,dtype=np.float64)),
                        'negative':sum(v<0 for v in finite),'zero':sum(v==0 for v in finite),
                        'positive':sum(v>0 for v in finite),'null':len(values)-len(finite)}
                pixel=[g['views'][view]['full_observed']['setting_contrasts'][contrast]['signed_pixel_map'] for g in selected]
                rows.append({'arm':arm,'view':view,'conditional_contrast':contrast,'cases':5,
                    'metric_differences':metrics,'signed_pixel_effect':{
                        'mean_absolute_RGB_unweighted_case_mean':float(np.mean([v['mean_absolute_RGB'] for v in pixel])),
                        'maximum_absolute_RGB_across5cases':max(v['maximum_absolute_RGB'] for v in pixel),
                        'mean_signed_luminance_unweighted_case_mean':float(np.mean([v['signed_mean_luminance'] for v in pixel]))}})
    return {'complete':True,'summaries':rows,'all15_groups_retained':True,
        'selection_is_purposive_after_prior_outputs':True,'three_photographs_not_fifteen_independent_people':True,
        'unweighted_case_means_not_population_or_unique_training_cause':True,
        'pixel_interactions_and_metric_interactions_are_distinct':True,
        'posthoc_route_or_best_setting_selected':False,'model_qualified':False,'quality_gates_changed':False}


def audit(folder,manifest_sha,worker_sha,out):
    import ast
    import time
    started=time.monotonic()
    require(not any(k in sys.modules for k in ('torch','tensorflow','jax')),'No neural framework in saved-artifact auditor')
    folder=Path(folder).resolve(strict=True);out=Path(out).absolute()
    require(folder.is_relative_to(ROOT/'outputs') and out.parent.resolve(strict=True).is_relative_to(ROOT/'outputs') and
        not out.exists(),'Exclusive new saved-artifact audit namespace required')
    prep=ROOT/'outputs/codeformer_filter_path_isolation_v1_preparation'
    contract=file_bytes(prep/'metric_contract_v1.json',METRIC_CONTRACT_SHA,4856)
    prefix=file_bytes(prep/'metric_helper_source_at_freeze_v1.py',HELPER_PREFIX_SHA,11930)
    require(Path(__file__).read_bytes().startswith(prefix),'Prospective frozen numerical helper prefix changed')
    manifest_data=file_bytes(folder/'manifest.json',manifest_sha,cap=2*1024**2);doc=unique_json(manifest_data)
    require(doc['format']=='codeformer-frozen-path-isolation-results-v1' and doc['complete'] is True and
        doc['source_sha256']==worker_sha and doc['trusted_helper_sha256']==OLD_HELPER_SHA and
        doc['protocol_sha256']=='23772988d16e11f52351365e66ce1b6d8ef77694f873966db98e24512b0aa86c' and
        doc['caps']==REVIEWED_CAPS and 0<=doc['seconds']<=600,'Completed exact frozen path manifest required')
    worker=file_bytes(ROOT/'scripts/run_codeformer_filter_path_isolation_v1.py',worker_sha)
    assignments={node.targets[0].id:node.value for node in ast.parse(worker).body
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)}
    require(ast.literal_eval(assignments['PROTOCOL_SHA'])==doc['protocol_sha256'] and
        ast.literal_eval(assignments['HELPER_SHA'])==OLD_HELPER_SHA,'Worker fixed protocol/helper source differs')
    # CAPS contains only inert arithmetic, which is evaluated without executing
    # the source module or importing any framework.
    def constant(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float,str):return node.value
        if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Mult,ast.Pow)):
            left,right=constant(node.left),constant(node.right)
            require(type(left) in (int,float) and type(right) in (int,float),'Numeric cap operands required')
            return left*right if isinstance(node.op,ast.Mult) else left**right
        raise ValueError('Unexpected cap AST')
    caps_node=assignments['CAPS'];require(isinstance(caps_node,ast.Dict),'Literal source cap dictionary required')
    require({constant(k):constant(v) for k,v in zip(caps_node.keys,caps_node.values)}==REVIEWED_CAPS,
        'Reviewed worker resource literals changed')
    files=doc['files'];require(len(files)+1<=170 and 'manifest.json' not in files,'Nonrecursive finite file manifest required')
    total=0
    for name,pin in files.items():
        canonical_relative(name);require(set(pin)=={'path','sha256','bytes'} and pin['path']==name,'Exact output pin schema')
        file_bytes(folder/name,pin['sha256'],pin['bytes']);total+=pin['bytes']
    actual_files=set();actual_dirs=set()
    for path in folder.rglob('*'):
        info=path.lstat();require(not stat.S_ISLNK(info.st_mode) and not getattr(info,'st_file_attributes',0)&1024,
            'Linked/reparse output namespace entry')
        if stat.S_ISREG(info.st_mode):actual_files.add(path.relative_to(folder).as_posix())
        elif stat.S_ISDIR(info.st_mode):actual_dirs.add(path.relative_to(folder).as_posix())
        else:raise ValueError('Nonordinary output namespace entry')
    expected_dirs={p.as_posix() for n in files for p in PurePosixPath(n).parents if p.as_posix()!='.'}
    require(actual_files==set(files)|{'manifest.json'} and actual_dirs==expected_dirs and
        total+len(manifest_data)<=64*1024**2,'Closed finite output file/directory/byte namespace required')
    def get(pin):
        require(pin==files.get(pin['path']),'Output pin not exactly manifest-bound')
        return file_bytes(folder/pin['path'],pin['sha256'],pin['bytes'])
    p=unique_json(get(files['protocol.json']))
    require(files['protocol.json']['sha256']==doc['protocol_sha256'] and
        get(files['protocol.json'])==file_bytes(prep/'protocol_v1.json',doc['protocol_sha256']) and
        tuple(c['id'] for c in p['cases'])==CASE_IDS and p['metric_analysis']['contract']['sha256']==METRIC_CONTRACT_SHA,
        'Exact frozen prospective protocol/case/metric contract required')
    require(p['caps']==dict(REVIEWED_CAPS,resource_sampling='aroundeachcall andindependent1s watchdog;sampledpeak limitations disclosed') and
        p['planned_counts']=={'models':3,'cases':5,'factorial_cells':60,'cached_w1_adain_true_cells':15,
        'new_factorial_cells':45,'fresh_clear_baseline_parity_calls':3,'total_new_forward_calls':48,
        'optimizer_updates':0,'backward_calls':0,'factorial_groups':15,'fresh_per_model_calls':16,'source_photographs':3},
        'Exact twelve-field prospective counts/caps required')
    metadata={}
    require(len(p['bound_metadata'])==14 and len(p['freshly_hashed_dependency_files'])==70,'Fixed metadata/dependency census')
    for key,pin in p['bound_metadata'].items():
        value=workspace_pin(pin)
        if pin['path'].endswith('.json'):metadata[key]=unique_json(value)
    for pin in p['freshly_hashed_dependency_files']:
        workspace_pin(pin,load=False,original=pin['path']=='outputs/codeformer_restoration_pretrained_v1/codeformer.pth')
    for pin in p['architecture_sources']:workspace_pin(pin,load=False)
    require(metadata['comparison_pixel_audit']['pass'] is True and metadata['comparison_journal_audit']['pass'] is True and
        p['models']==metadata['model_binding']['models'],'Previous complete comparison/model binding differs')
    bound_models={arm:next(m for m in p['models'] if m['arm']==MODELS[arm][0]) for arm in ARMS}
    model_file_proofs={}
    for arm,m in bound_models.items():
        require(m['tensor_count']==515 and m['canonical515_tensor_sha256']==MODELS[arm][2] and
            m['file']['sha256']==MODELS[arm][1] and m['all515finite_float32_CPU'] is True,'Exact retained515 model binding required')
        model_pin=dict(m['file'],absolute_path=str(ROOT/m['file']['path']))
        workspace_pin(model_pin,load=False,original=arm=='original0')
        model_file_proofs[arm]={'path':model_pin['path'],'sha256':model_pin['sha256'],'bytes':model_pin['bytes'],
            'freshly_stream_hashed_after_inference':True,'hardlink_count':3 if arm=='original0' else 1,
            'tensor_checkpoint_deserialized_or_executed':False}
    source_contract=unique_json(get(files['source_contract.json']))
    require(source_contract['source_sha256']==worker_sha and source_contract['trusted_helper_sha256']==OLD_HELPER_SHA and
        source_contract['caps']==REVIEWED_CAPS and source_contract['new_factorial_outputs']==45 and
        source_contract['fresh_clear_parity_outputs']==3 and source_contract['cached_cells_reused']==15 and
        source_contract['hardcoded_settings_only']==[SETTING_VALUES[s] for s in SETTINGS] and
        source_contract['no_training_scoring_or_app_change'] is True and source_contract['no_extra_forwards_for_readouts'] is True and
        source_contract['internal_token_readout_is_recomputed_not_actual_lookup'] is True,'Exact source execution contract differs')
    runtime=unique_json(get(files['runtime.json']));previous_runtime=metadata['comparison_runtime']
    require(runtime['device']=='cpu' and runtime['threads']==4 and runtime['interop_threads']==1 and
        all(runtime[key]==previous_runtime[key] for key in ('torch','NumPy','Pillow')),'Exact retained-baseline CPU runtime differs')
    validate_resource(runtime['resources'])
    state_doc=unique_json(get(files['model_state_proofs.json']))
    require(state_doc['complete'] is True and state_doc['models']==doc['model_proofs'] and len(doc['model_proofs'])==3 and
        {r['arm'] for r in doc['model_proofs']}==set(ARMS),'Exact three distinct frozen-state proofs required')
    proofs={r['arm']:r for r in doc['model_proofs']}
    names=sorted(n for n in metadata['model_binding']['full515_tensor_schema']
        if n.startswith(('generator.','quantize.')))
    require(len(names)==165 and sum(n.startswith('generator.') for n in names)==164 and
        sum(n.startswith('quantize.') for n in names)==1,'Exact165 shared generator/codebook schema required')
    shared=doc['shared165_generator_codebook_identical_across_three_models']
    require(set(shared)=={'tensors','generator_tensors','codebook_tensors','canonical_sha256','tensor_names'} and
        shared['tensors']==165 and shared['generator_tensors']==164 and shared['codebook_tensors']==1 and
        shared['tensor_names']==names and sha_string(shared['canonical_sha256']),
        'Shared165 source-bound subset digest/namespace differs')
    for arm,proof in proofs.items():
        require(proof['before_state_sha256']==proof['after_state_sha256']==MODELS[arm][2] and
            proof['file']==bound_models[arm]['file'] and proof['top_level_calls']==16 and
            proof['fusion_calls']==dict.fromkeys(FUSIONS,6) and proof['shared_generator_codebook']==shared,
            'Frozen model state/shared prior/hook totals differ')
    cases=p['cases'];previous_cases={c['id']:c for c in metadata['comparison_manifest']['cases']}
    inputs={};targets={};regions={};cache={};raw_bank={};png_bank={};readout_bank={}
    for case in cases:
        key=case['id'];require(case['comparison_case']==previous_cases[key] and
            case['input_review']==previous_cases[key]['input_review'] and case['appearance_review_required'] is False and
            case['input_review']['decision'] in ('readable','borderline') and case['optimization_permitted_now'] is False and
            case['DEV_exclusion_preserved'] is True,'Prospective selection/review/split/admission changed')
        inputs[key]=rgb_png(workspace_pin(case['input']));targets[key]=rgb_png(workspace_pin(case['targets']['clean256']))
        regions[key]=region_masks(key,footprint_png(workspace_pin(case['corruption_footprint'])))
        require(set(case['cached_baseline'])==set(ARMS),'All fifteen cached baseline cells required')
        for arm,bank in case['cached_baseline'].items():
            raw=raw_rgb(workspace_pin(bank['raw']));png=rgb_png(workspace_pin(bank['png']));check_floor(raw,png)
            e=unique_json(workspace_pin(bank['evidence']))
            require(e['id']==key and e['arm']==arm and e['input_review']==case['input_review'] and
                e['canonical515_state_sha256']==MODELS[arm][2] and
                digest(raw.tobytes())==bank['array_digests']['RAW_array_sha256'] and
                digest(png.tobytes())==bank['array_digests']['PNG_RGB8_sha256'],'Cached exact RAW/PNG/evidence lineage differs')
            cache[arm,key]=(raw,png);raw_bank[arm,key,'w1_adain1']=raw;png_bank[arm,key,'w1_adain1']=png
    require(len(doc['cells'])==60 and len(doc['fresh_clear_parity'])==3,'Exact60 cells and three separate parity outputs required')
    cells={(r['arm'],r['id'],r['setting']['id']):r for r in doc['cells']}
    require(len(cells)==60 and set(cells)=={(arm,key,s) for arm in ARMS for key in CASE_IDS for s in SETTINGS},
        'Unique complete factorial cell grid required')
    expected_files={'protocol.json','source_contract.json','runtime.json','model_state_proofs.json','journal.jsonl'}
    evidence_by_call={};parity_by_arm={};readout_rows=[]
    for row in [*doc['fresh_clear_parity'],*doc['cells']]:
        arm,key,setting=row['arm'],row['id'],row['setting']['id'];kind=row['kind']
        require(row['setting']==SETTING_VALUES[setting],'Exact model-cell setting required')
        if kind=='cached_prior_comparison':
            require(setting=='w1_adain1' and row['cached']==next(c for c in cases if c['id']==key)['cached_baseline'][arm],
                'Cached factorial cell not exact prospective baseline pin')
            continue
        require(kind in ('fresh_clear_parity','new_factorial') and arm in ARMS and key in CASE_IDS and
            ((kind=='fresh_clear_parity' and key==CASE_IDS[0] and setting=='w1_adain1') or
             (kind=='new_factorial' and setting in SETTINGS[:3])),'Exact fresh parity/new factorial cell role required')
        prefix_path=('parity/' if kind=='fresh_clear_parity' else 'outputs/')+arm+'/'+key+'/'+setting+'/'
        for name,suffix in [('raw','raw.npy'),('png','native.png'),('evidence','evidence.json')]:
            require(row[name]['path']==prefix_path+suffix,'Exact per-call output namespace required');expected_files.add(row[name]['path'])
        raw=raw_rgb(get(row['raw']));png=rgb_png(get(row['png']));check_floor(raw,png)
        e=unique_json(get(row['evidence']));case=next(c for c in cases if c['id']==key)
        require(e['format']=='codeformer-frozen-path-output-v1' and e['id']==key and e['reference_id']==case['reference_id'] and
            e['split']==case['split'] and e['profile']==case['profile'] and e['input_review']==case['input_review'] and
            e['appearance_review_required'] is False and e['input']==case['input'] and e['targets_not_scored']==case['targets'] and
            e['arm']==arm and e['setting']==row['setting'] and e['kind']==kind and e['model_binding']==bound_models[arm] and
            e['state_sha256']==MODELS[arm][2],'Exact per-return source/model/setting lineage differs')
        require(e['raw']==row['raw'] and e['png']==row['png'] and e['RAW_array_sha256']==digest(raw.tobytes()) and
            e['PNG_RGB8_sha256']==digest(png.tobytes()) and e['clear_parity']==row['clear_parity'],
            'Per-return saved array/PNG fingerprint differs')
        require(e['device']=='cpu' and e['eval_all_modules'] is e['no_grad'] is True and
            e['parameters_require_grad'] is False and e['internal_resolution']==512 and e['output_resolution']==256 and
            e['geometry_changed'] is False and e['alignment']==e['display_processing']=='none' and
            e['optimizer_updates']==e['backward_calls']==0 and e['paired_metrics'] is None and e['model_qualified'] is False and
            type(e['seconds']) in (int,float) and math.isfinite(e['seconds']) and 0<=e['seconds']<=600,
            'Frozen inference geometry/finite timing/no qualification policy differs')
        observed_setting(e['observer'],setting);signature=readout_signature(e['internal_readout'])
        require(e['same_upstream_within_input_model'] is True,'Reported internal route equality failed')
        require(type(e['call']) is int and 1<=e['call']<=48 and e['call'] not in evidence_by_call,'Unique bounded forward call required')
        e['_pin']=row['evidence'];evidence_by_call[e['call']]=e
        readout_bank[arm,key,setting]=signature
        readout_rows.append({'arm':arm,'id':key,'setting':setting,'kind':kind,'call':e['call'],
            'evidence':row['evidence'],'internal_readout':e['internal_readout'],'observer':e['observer']})
        if kind=='fresh_clear_parity':
            require(arm not in parity_by_arm,'Unique fresh clear parity per model required')
            old_raw,old_png=cache[arm,key]
            actual={'RAW_exact':bool(np.array_equal(raw,old_raw)),'PNG_exact':bool(np.array_equal(png,old_png)),
                'RAW_maximum_error':float(np.abs(raw.astype(np.float64)-old_raw.astype(np.float64)).max()),
                'PNG_maximum_error':int(np.abs(png.astype(np.int16)-old_png.astype(np.int16)).max())}
            require(actual==row['clear_parity']=={'RAW_exact':True,'PNG_exact':True,'RAW_maximum_error':0.,'PNG_maximum_error':0},
                'Fresh clear baseline bit-exact parity failed')
            parity_by_arm[arm]=actual
        else:
            require(row['clear_parity'] is None,'New intervention must not masquerade as fresh cached parity')
            raw_bank[arm,key,setting]=raw;png_bank[arm,key,setting]=png
    require(set(files)==expected_files and len(files)==149 and len(evidence_by_call)==48 and
        set(parity_by_arm)==set(ARMS),'Exact149 nonrecursive files/48 calls/three clear parity closures required')
    for arm in ARMS:
        for key in CASE_IDS:
            values=[readout_bank[arm,key,s] for s in SETTINGS[:3]]
            if key==CASE_IDS[0]:values.append(readout_bank[arm,key,'w1_adain1'])
            require(all(v==values[0] for v in values),'Returned logits/LQ/tokens differ across declared late-route settings')
    journal_payload=get(files['journal.jsonl']);require(len(journal_payload)<=1024**2,'Bounded journal required')
    journal=[unique_json(line) for line in journal_payload.splitlines()]
    resources=[runtime['resources'],*validate_journal(journal,evidence_by_call,proofs)]
    require(len(resources)==101,'Exact101 source-bound resource samples required')
    require(doc['actual_new_forward_calls']==48 and doc['new_factorial_outputs']==45 and doc['cached_cells_reused']==15 and
        doc['fresh_clear_parity_outputs']==3 and doc['actual_optimizer_updates']==doc['actual_backward_calls']==0 and
        doc['target_scoring'] is False and doc['same_upstream_readouts_all_inputs_models'] is True and
        doc['settings_are_global_late_path_interventions_not_additive_individual_fusion_layers'] is True and
        doc['recomputed_token_readout_not_direct_internal_lookup_capture'] is True and
        doc['exploratory_after_known_failures_not_unseen_validation'] is True and
        doc['all96_previous_inputs_and24_refusals_unchanged'] is True and doc['app_or_model_changed'] is False and
        doc['new_acceptance_thresholds'] is False and doc['model_qualified'] is False,
        'Authoritative finite inference outcome/claim boundary differs')
    groups=[];control=[];max_closure=0.
    for case in cases:
        key=case['id'];clean=targets[key].astype(np.float64)/255.
        control.append({'id':key,'regions':{region:{'raw':residual_metrics(inputs[key].astype(np.float32)/np.float32(255),clean,mask),
            'png':residual_metrics(inputs[key].astype(np.float64)/255.,clean,mask)} for region,mask in regions[key].items()},
            'input_control_is_not_claimed_RAW_floor_model_output':True})
        for arm in ARMS:
            views={'raw':factorial_diagnostics({s:raw_bank[arm,key,s] for s in SETTINGS},clean,regions[key]),
                'png':factorial_diagnostics({s:png_bank[arm,key,s].astype(np.float64)/255. for s in SETTINGS},clean,regions[key])}
            for view in views.values():
                for region in view.values():
                    max_closure=max(max_closure,max(abs(m['MSE_decomposition_closure_residual'])
                        for m in region['paired_setting_metrics'].values()))
            groups.append({'id':key,'reference_id':case['reference_id'],'split':case['split'],'profile':case['profile']['id'],
                'arm':arm,'decision':case['input_review']['decision'],'optimization_permitted_now':False,
                'target256':case['targets']['clean256'],'views':views})
    require(len(groups)==15 and len(raw_bank)==len(png_bank)==60,'Exact15 groups/60 complete factorial cells required')
    require(not any(k in sys.modules for k in ('torch','tensorflow','jax')),'Auditor imported neural framework')
    out.mkdir();sheets=render_sheets(out,cases,inputs,targets,png_bank)
    artifacts=[]
    artifacts.append(save_json_exclusive(out/'factorial_metrics_v1.json',{'complete':True,'metric_contract_sha256':METRIC_CONTRACT_SHA,
        'groups':groups,'resize_control':control,'factorial_cells':60,'fresh_parity_excluded_from_metric_denominator':True,
        'native_CCTV_cases':0,'FINAL_cases':0,'model_qualified':False,'new_acceptance_thresholds':False}))
    artifacts.append(save_json_exclusive(out/'conditional_effects_summary_v1.json',aggregate_conditional_effects(groups)))
    artifacts.append(save_json_exclusive(out/'readout_audit_v1.json',{'complete':True,'returns':sorted(readout_rows,key=lambda r:r['call']),
        'new_top_level_calls':48,'four_fusion_calls':72,'per_fusion_site_calls':dict.fromkeys(FUSIONS,18),
        'all15_groups_new3settings_exact_logits_LQ_tokens':True,'clear3_groups_new4settings_exact':True,
        'other12_cached_baselines_have_no_captured_internals':True,'tokens_recomputed_not_direct_lookup_capture':True,
        'internal_hashes_source_bound_not_independently_replayed':True,'confidence_or_identity_calibration':False}))
    artifacts.append(save_json_exclusive(out/'visual_manifest_v1.json',sheets))
    receipt={'format':'codeformer-filter-path-isolation-independent-saved-artifact-audit-v1','complete':True,'pass':True,
        'UTC':datetime.now(timezone.utc).isoformat(),'worker_manifest':{'path':str(folder/'manifest.json'),'sha256':manifest_sha,'bytes':len(manifest_data)},
        'worker_sha256':worker_sha,'auditor_sha256':digest(Path(__file__).read_bytes()),'protocol_sha256':doc['protocol_sha256'],
        'metric_contract_sha256':METRIC_CONTRACT_SHA,'frozen_metric_helper_prefix_sha256':HELPER_PREFIX_SHA,
        'files':artifacts,'freshly_hashed_dependency_files':70,'bound_metadata':14,'architecture_sources':3,
        'model_files_freshly_stream_hashed_after_inference':model_file_proofs,
        'closed_worker_files':150,'closed_worker_bytes':total+len(manifest_data),'new_forwards_reserved_returned_committed':48,
        'new_factorial_cells':45,'cached_baseline_cells':15,'factorial_groups':15,'factorial_cells':60,
        'RAW_PNG_new_triples':48,'all48_exact_RAW_geometry_finite_floor_PNG':True,'all15_cached_triples_exact_hash_floor':True,
        'fresh_clear_parity_bit_exact':parity_by_arm,'durable_journal_records':154,'frozen515_before_after_exact_models':3,
        'shared165_generator_codebook_source_bound_digest':shared,
        'shared165_values_not_independently_deserialized_or_replayed_by_this_auditor':True,
        'observed_top_level_calls':48,'observed_total_fusion_calls':72,'observed_fusion_calls_per_site':dict.fromkeys(FUSIONS,18),
        'readouts_recomputed_from_returned_logits_not_actual_lookup_capture':True,
        'upstream_equality_scope':'All fifteen groups new three settings; only three clear groups include fresh fourth setting. No old nonclear internals invented.',
        'resources':{'samples':len(resources),'maximum_recorded_RSS_bytes':max(r['process_RSS_bytes'] for r in resources),
            'minimum_recorded_RAM_available_bytes':min(r['available_RAM_bytes'] for r in resources),
            'minimum_recorded_disk_free_bytes':min(r['disk_free_bytes'] for r in resources),
            'recorded_samples_not_independent_continuous_watchdog_profile':True},
        'exact256_sheet_panels':90,'sheets':5,'all5_root_visual_review_pending':True,
        'maximum_absolute_MSE_identity_roundoff_not_gate':max_closure,'useful_brightening_remains_desired':True,
        'conditional_forward_effects_not_unique_training_cause':True,'metric_DID_not_metric_of_signed_pixel_interaction':True,
        'no_metric_selected_default_route_or_new_threshold':True,'all96_72_core63_flag9_and24_skips_unchanged':True,
        'new_NN_backward_optimizer_VM_browser_calls':0,'model_checkpoint_deserialized':False,'worker_source_imported':False,
        'native_CCTV_cases':0,'FINAL_cases':0,'model_qualified':False,'goal_complete':False,'seconds':time.monotonic()-started}
    return save_json_exclusive(out/'audit_v1.json',receipt)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--manifest-sha',required=True)
    parser.add_argument('--worker-sha',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.folder,args.manifest_sha,args.worker_sha,args.out)),flush=True)

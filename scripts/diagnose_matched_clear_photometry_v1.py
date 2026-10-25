"""Descriptive changes of eight saved TRAIN clear faces; never restore/train.

Signed maps are explicitly amplified diagnostics, not generated face results.
The analytic global/spatial attribution does not establish a training remedy.
"""
from datetime import datetime, timezone
import hashlib
import importlib.abc
import importlib.util
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_matched_feature_learning_v1_return_audit_v1'
STATIC = ROOT / 'outputs/cctv_codeformer_matched_feature_learning_v1_vm'
RETAINED = OUT / 'retained/cctv_codeformer_matched_feature_learning_v1_vm'
RUN = RETAINED / 'outputs/matched_feature_learning_v1'
LUMA = (.2126, .7152, .0722)


class Blocker(importlib.abc.MetaPathFinder):
    attempted = []
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','torchvision','tensorflow','jax','jaxlib','keras'} or fullname.endswith(('.worker','.model')):
            self.attempted.append(fullname)
            raise RuntimeError('Forbidden photometric diagnosis import: ' + fullname)
        return None


def need(ok, message):
    if not ok: raise ValueError(message)


def block_mean(value, *, np):
    need(isinstance(value,np.ndarray) and value.shape==(256,256,3) and np.isfinite(value).all(),
         'Finite RGB256 diagnostic field required')
    return value.astype(np.float64).reshape(16,16,16,16,3).mean(axis=(1,3))


def symmetric_attribution(base, delta, target, metric, *, np):
    """Analytic additive delta attribution, no clipping or image synthesis.

    Global RGB delta is its channelwise spatial mean; the rest is zero-mean.
    Symmetric ordering splits their interaction equally. This descriptive
    allocation is defined by this chosen decomposition, not a causal remedy.
    """
    original=base.astype(np.float64);change=delta.astype(np.float64);reference=target.astype(np.float64)
    global_delta=change.mean(axis=(0,1),keepdims=True);spatial_delta=change-global_delta
    f0=metric(original-reference);fg=metric(original+global_delta-reference)
    fs=metric(original+spatial_delta-reference);fall=metric(original+change-reference)
    global_part=.5*((fg-f0)+(fall-fs));spatial_part=.5*((fs-f0)+(fall-fg))
    need(abs((global_part+spatial_part)-(fall-f0))<1e-14,'Exact analytic attribution identity failed')
    return {'initial':f0,'current':fall,'delta_positive_worse':fall-f0,
            'global_channel_mean_delta_attribution':global_part,'remaining_spatial_delta_attribution':spatial_part,
            'symmetric_ordering_interaction_split':True,'counterfactual_values_are_unclipped_scalar_diagnostics_only':True}


def signed_map(value, scale, *, np):
    need(isinstance(value,np.ndarray) and value.shape==(256,256) and np.isfinite(value).all() and scale>0,
         'Finite scalar256 map and positive fixed scale required')
    normalized=np.clip(value/scale,-1.,1.);magnitude=np.abs(normalized)
    white=np.full((256,256,3),255.,np.float64)
    positive=np.asarray([210.,38.,38.]);negative=np.asarray([38.,80.,210.])
    color=np.where((normalized>=0)[...,None],positive,negative)
    return np.floor(white*(1-magnitude[...,None])+color*magnitude[...,None]+.5).astype(np.uint8)


def integer_error_decomposition(initial,current,target,*,np):
    for value in(initial,current,target):
        need(isinstance(value,np.ndarray) and value.shape==(256,256,3) and value.dtype==np.uint8,
             'Exact RGB8 arrays required for integer decomposition')
    before=initial.astype(np.int16);after=current.astype(np.int16);truth=target.astype(np.int16)
    movement=after-before;difference=np.abs(after-truth)-np.abs(before-truth)
    changed=movement!=0
    need(not np.any(difference[~changed]),'Unchanged RGB component cannot change integer target error')
    better=int(difference[difference<0].sum(dtype=np.int64));worse=int(difference[difference>0].sum(dtype=np.int64))
    net=int(difference.sum(dtype=np.int64));need(net==better+worse,'Integer contribution sum differs')
    by_channel=[]
    for channel,name in enumerate(('R','G','B')):
        d=difference[:,:,channel];m=movement[:,:,channel]
        by_channel.append({'channel':name,'net_target_error_code_units':int(d.sum(dtype=np.int64)),
            'improved_error_code_units':int(d[d<0].sum(dtype=np.int64)),
            'worsened_error_code_units':int(d[d>0].sum(dtype=np.int64)),
            'changed_components':int(np.count_nonzero(m)),
            'signed_output_code_sum':int(m.sum(dtype=np.int64)),
            'brightened_component_error_contribution':int(d[m>0].sum(dtype=np.int64)),
            'darkened_component_error_contribution':int(d[m<0].sum(dtype=np.int64))})
    need(sum(row['net_target_error_code_units']for row in by_channel)==net,'Channel integer sum differs')
    return {'net_target_error_code_units':net,'improved_error_code_units':better,'worsened_error_code_units':worse,
        'changed_components':int(np.count_nonzero(changed)),
        'error_improved_components':int(np.count_nonzero(difference<0)),
        'error_worsened_components':int(np.count_nonzero(difference>0)),
        'changed_but_equal_error_components':int(np.count_nonzero(changed&(difference==0))),
        'maximum_absolute_output_code_change':int(np.abs(movement).max()),
        'brightened_component_error_contribution':int(difference[movement>0].sum(dtype=np.int64)),
        'darkened_component_error_contribution':int(difference[movement<0].sum(dtype=np.int64)),
        'unchanged_component_error_contribution':0,'by_channel':by_channel},difference,movement


def field_diagnosis(initial,current,target,*,np):
    start=initial.astype(np.float64);now=current.astype(np.float64);truth=target.astype(np.float64)
    delta=now-start;res0=start-truth;res1=now-truth
    mean=delta.mean(axis=(0,1));spatial=delta-mean
    energy=float(np.square(delta).mean());constant_energy=float(np.square(mean).mean());spatial_energy=float(np.square(spatial).mean())
    need(abs(energy-constant_energy-spatial_energy)<1e-15,'Constant/spatial energy identity differs')
    b0,b1,bd=(block_mean(x,np=np)for x in(res0,res1,delta))
    block_error_change=np.abs(b1).mean(axis=2)-np.abs(b0).mean(axis=2)
    block_energy=np.square(delta).reshape(16,16,16,16,3).mean(axis=(1,3,4))
    ordered=np.sort(block_energy.reshape(-1))[::-1]
    top_worse=np.argsort(block_error_change.reshape(-1))[::-1][:8]
    result={'signed_RGB_mean_residual_initial':res0.mean(axis=(0,1)).tolist(),
        'signed_RGB_mean_residual_current':res1.mean(axis=(0,1)).tolist(),
        'signed_RGB_mean_output_shift':mean.tolist(),
        'signed_encoded_luminance_mean_shift':float(mean@np.asarray(LUMA)),
        'mean_absolute_RGB_output_change':float(np.abs(delta).mean()),
        'mean_square_output_delta':energy,'constant_channel_mean_delta_energy':constant_energy,
        'remaining_zero_mean_spatial_delta_energy':spatial_energy,
        'constant_mean_fraction_of_output_delta_energy':constant_energy/energy if energy else 0.,
        'top26_of256_blocks_fraction_of_output_delta_energy':float(ordered[:26].sum()/ordered.sum())if ordered.sum()else 0.,
        'mean_abs_block_delta_RGB':np.abs(bd).mean(axis=(0,1)).tolist(),
        'coarse_RGB_error_initial':float(np.abs(b0).mean()),'coarse_RGB_error_current':float(np.abs(b1).mean()),
        'coarse_RGB_error_delta':float(np.abs(b1).mean()-np.abs(b0).mean()),
        'coarse_error_blocks_worse':int(np.count_nonzero(block_error_change>0)),
        'coarse_error_blocks_better':int(np.count_nonzero(block_error_change<0)),
        'coarse_error_blocks_equal':int(np.count_nonzero(block_error_change==0)),
        'largest8_coarse_block_error_increases':[{'block_row':int(x//16),'block_column':int(x%16),
            'pixel_rectangle':[int(x%16)*16,int(x//16)*16,int(x%16+1)*16,int(x//16+1)*16],
            'error_delta':float(block_error_change.reshape(-1)[x]),'RGB_mean_residual_before':b0.reshape(-1,3)[x].tolist(),
            'RGB_mean_residual_after':b1.reshape(-1,3)[x].tolist(),'RGB_mean_output_shift':bd.reshape(-1,3)[x].tolist()}
            for x in top_worse],
        'raw_MAE_attribution':symmetric_attribution(start,delta,truth,lambda r:float(np.abs(r).mean()),np=np),
        'coarse_RGB_error_attribution':symmetric_attribution(start,delta,truth,lambda r:float(np.abs(block_mean(r,np=np)).mean()),np=np)}
    return result,delta,np.abs(res1).mean(axis=2)-np.abs(res0).mean(axis=2)


def main():
    begun=time.monotonic();blocker=Blocker();sys.meta_path.insert(0,blocker)
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    source=ROOT/'scripts/audit_matched_feature_return_pixels_v1.py'
    spec=importlib.util.spec_from_file_location('verified_saved_array_primitives',source)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    consumed={}
    def pin(path):
        need(time.monotonic()-begun<300,'Finite300s descriptive diagnosis stop')
        path=helper.ordinary(path);raw=path.read_bytes();value={'path':path.relative_to(ROOT).as_posix(),
            'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        prior=consumed.get(value['path']);need(prior is None or prior==value,'Consumed diagnostic input changed')
        consumed[value['path']]=value;return value
    def doc(path):pin(path);return helper.strict_json(path.read_bytes())
    pixels=doc(OUT/'pixels.json');transport=doc(OUT/'transport.json');index=doc(OUT/'member_index.json');p=doc(STATIC/'protocol.json')
    need(pixels['complete']and transport['complete']and pin(source)==pixels['source']and
         pin(OUT/'member_index.json')==transport['member_index'],'Completed exact independent saved-pixel evidence required')
    def checked(base,binding):
        path=base/binding['path'];actual=pin(path)
        need({k:actual[k]for k in('bytes','sha256')}=={k:binding[k]for k in('bytes','sha256')},'Actual bound diagnostic file differs')
        if base==RETAINED:need(index['members'][binding['path']]=={k:actual[k]for k in('bytes','sha256')},'Archive-bound output required')
        return path
    def rgb(base,binding):
        with Image.open(checked(base,binding))as image:
            need(image.mode=='RGB'and image.size==(256,256),'Exact RGB256 required');return np.asarray(image).copy()
    def raw(binding):return helper.decode_array_bytes(checked(RETAINED,binding).read_bytes(),np=np)
    records={}
    for path in sorted((RUN/'calls').glob('*/record.json')):
        r=doc(path);actual=pin(path);need(index['members'][path.relative_to(RETAINED).as_posix()]==
            {k:actual[k]for k in('bytes','sha256')},'Exact source call metadata required')
        call=r['call'];records[(call['arm'],call['bank'],call['id'],call['phase'],call['new_update'])]=r
    cases=[];maps=[];target_by={};integer_net={'A':0,'B':0}
    for row in p['selection']['new_monitors']:
        if row['role']!='clear':continue
        cid=row['id'];photo=rgb(STATIC,row['target']);target=photo.astype(np.float32)/np.float32(255);target_by[cid]=photo
        baseline=records[('A','new_monitors',cid,'initial_quality',0)];r0,q0=raw(baseline['outputs']['raw']),rgb(RETAINED,baseline['outputs']['png'])
        entry={'reference_id':row['reference_id'],'case_key':'new_monitors:'+cid,'target':row['target'],
               'initial_u64_outputs':baseline['outputs'],'arms':{}}
        rowmaps={}
        for arm in('A','B'):
            current=records[(arm,'new_monitors',cid,'quality_snapshot',8)];r1,q1=raw(current['outputs']['raw']),rgb(RETAINED,current['outputs']['png'])
            photo_diag,delta,error=field_diagnosis(r0,r1,target,np=np)
            integer,error_codes,movement=integer_error_decomposition(q0,q1,photo,np=np);integer_net[arm]+=integer['net_target_error_code_units']
            png0=q0.astype(np.float32)/np.float32(255);png1=q1.astype(np.float32)/np.float32(255)
            png_metric0=float(np.abs(png0.astype(np.float64)-target.astype(np.float64)).mean());png_metric1=float(np.abs(png1.astype(np.float64)-target.astype(np.float64)).mean())
            png_residual0=png0.astype(np.float64)-target.astype(np.float64)
            png_residual1=png1.astype(np.float64)-target.astype(np.float64)
            need(photo_diag['raw_MAE_attribution']['initial']==baseline['metrics']['raw']['mae']and
                 photo_diag['raw_MAE_attribution']['current']==current['metrics']['raw']['mae']and
                 png_metric0==baseline['metrics']['native_png']['mae']and png_metric1==current['metrics']['native_png']['mae'],
                 'Descriptive decomposition must equal exact independently recomputed gate metrics')
            entry['arms'][arm]={'current_outputs':current['outputs'],'raw':photo_diag,'floor_PNG_integer_error_decomposition':integer,
                'PNG_MAE_initial':png_metric0,'PNG_MAE_current':png_metric1,'PNG_MAE_delta':png_metric1-png_metric0,
                'PNG_signed_RGB_mean_residual_initial':png_residual0.mean(axis=(0,1)).tolist(),
                'PNG_signed_RGB_mean_residual_current':png_residual1.mean(axis=(0,1)).tolist(),
                'PNG_coarse_RGB_error_initial':float(np.abs(block_mean(png_residual0,np=np)).mean()),
                'PNG_coarse_RGB_error_current':float(np.abs(block_mean(png_residual1,np=np)).mean()),
                'PNG_mean_abs_block_output_delta_RGB':np.abs(block_mean(png1.astype(np.float64)-png0.astype(np.float64),np=np)).mean(axis=(0,1)).tolist(),
                'normalization_quantization_change_of_error_delta':(png_metric1-png_metric0)-photo_diag['raw_MAE_attribution']['delta_positive_worse']}
            rowmaps[arm]={'luma_delta':delta@np.asarray(LUMA),'raw_error_delta':error,'PNG_error_delta_codes':error_codes.mean(axis=2)}
        cases.append(entry);maps.append((cid,rowmaps))
    need(len(cases)==8 and integer_net['B']==1126,'Exact8 clear case scope and independently counted B PNG error delta required')
    # One clearly labeled diagnostic image; source photos remain unmodified.
    fontpath=Path('C:/Windows/Fonts/arial.ttf');font=ImageFont.truetype(str(fontpath),14)if fontpath.exists()else ImageFont.load_default()
    small=ImageFont.truetype(str(fontpath),12)if fontpath.exists()else ImageFont.load_default()
    width=5*264+8;top=120;row_height=308
    canvas=Image.new('RGB',(width,top+len(cases)*row_height+24),(245,243,238));draw=ImageDraw.Draw(canvas)
    draw.text((8,7),'DIAGNOSTIC RESIDUAL MAPS ONLY — NOT RESTORED FACE OUTPUTS',font=font,fill=(20,20,20))
    draw.text((8,30),'All 8 adaptive TRAIN clear faces; u64 baseline vs A8/B8. Photographic references at left; residual maps amplified.',font=small,fill=(30,30,30))
    draw.text((8,49),'RED = brighter / greater target error. BLUE = darker / less target error. WHITE = zero. Fixed scales; no per-photo normalization.',font=small,fill=(30,30,30))
    draw.text((8,68),'A/B signed encoded-luma delta: +/- 0.5 RGB code. B RAW error delta: +/- 0.5 code. B PNG error delta: +/- 1/3 code.',font=small,fill=(30,30,30))
    draw.text((8,87),'No generated correction, identity recovery, causal remedy, native CCTV or FINAL claim. Numeric attribution is defined, descriptive only.',font=small,fill=(30,30,30))
    cells=[];scales={'luma_delta':.5/255.,'raw_error_delta':.5/255.,'PNG_error_delta_codes':1/3.}
    for i,(case,(cid,rowmaps))in enumerate(zip(cases,maps)):
        y=top+i*row_height;ref=case['reference_id'];b=case['arms']['B'];mean_fraction=b['raw']['constant_mean_fraction_of_output_delta_energy']
        draw.text((8,y),ref+' | B PNG MAE delta '+format(b['PNG_MAE_delta'],'.9g')+' | B constant-RGB share of delta energy '+format(mean_fraction*100,'.2f')+'%',font=font,fill=(20,20,20))
        visual=[('Photographic TRAIN reference',target_by[cid]),
                ('A8 - u64: signed encoded luma',signed_map(rowmaps['A']['luma_delta'],scales['luma_delta'],np=np)),
                ('B8 - u64: signed encoded luma',signed_map(rowmaps['B']['luma_delta'],scales['luma_delta'],np=np)),
                ('B RAW target-error change',signed_map(rowmaps['B']['raw_error_delta'],scales['raw_error_delta'],np=np)),
                ('B PNG target-error change',signed_map(rowmaps['B']['PNG_error_delta_codes'],scales['PNG_error_delta_codes'],np=np))]
        for col,(label,array)in enumerate(visual):
            x=8+264*col;draw.text((x,y+21),label,font=small,fill=(25,25,25));canvas.paste(Image.fromarray(array),(x,y+41))
            need(np.array_equal(np.asarray(canvas.crop((x,y+41,x+256,y+297))),array),'Diagnostic pasted cell differs')
            cells.append({'reference_id':ref,'label':label,'rectangle':[x,y+41,x+256,y+297],
                'RGB8_sha256':hashlib.sha256(array.tobytes()).hexdigest(),'is_amplified_diagnostic_map':col!=0})
    visual=OUT/'photometric_diagnostic_v1.png'
    with visual.open('xb')as f:canvas.save(f,format='PNG')
    with Image.open(visual)as image:need(np.array_equal(np.asarray(image),np.asarray(canvas)),'Full diagnostic visual readback differs')
    aggregate={}
    for arm in('A','B'):
        rows=[row['arms'][arm]for row in cases];at=[r['raw']['raw_MAE_attribution']for r in rows];ct=[r['raw']['coarse_RGB_error_attribution']for r in rows]
        aggregate[arm]={'eight_equal_photograph_mean_raw_MAE_delta':sum(r['delta_positive_worse']for r in at)/8,
            'mean_raw_MAE_global_component_attribution':sum(r['global_channel_mean_delta_attribution']for r in at)/8,
            'mean_raw_MAE_remaining_spatial_attribution':sum(r['remaining_spatial_delta_attribution']for r in at)/8,
            'mean_coarse_error_global_component_attribution':sum(r['global_channel_mean_delta_attribution']for r in ct)/8,
            'mean_coarse_error_remaining_spatial_attribution':sum(r['remaining_spatial_delta_attribution']for r in ct)/8,
            'net_integer_floor_PNG_target_error_code_units':integer_net[arm]}
    report={'complete':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'status':'DESCRIPTIVE_SAVED_PIXEL_PHOTOMETRIC_DIAGNOSIS_NO_REMEDY_OR_GATE_CHANGE',
        'source':pin(Path(__file__)),'pixel_audit':pin(OUT/'pixels.json'),'photographs':8,'case_rows':cases,'aggregate':aggregate,
        'visualization':pin(visual),'visualization_cells':cells,'fixed_visual_scales':scales,
        'visual_clipping_policy':'Only diagnostic colors saturate beyond declared symmetric scales; no RAW/PNG image or metric is clipped/changed by the audit.',
        'signed_residual_policy':'RGB and encoded-luminance values are encoded sRGB-domain arithmetic, not physical luminance or calibrated lighting.',
        'decomposition_policy':'Delta=currentRAW-initialRAW; global term is per-channel image mean, spatial term has zero mean. Squared-energy split is orthogonal; MAE/coarse-error split is symmetric two-order analytic attribution without clipping. Neither is unique causal evidence.',
        'gate_status_immutable':'Both arms stopped at8. Coarse color checks are descriptive at8 and bind only unexecuted24/48. Do not infer future success or bypass failed PNG clear preservation.',
        'next_testable_hypothesis':'If the returned maps and independent review confirm spatially concentrated changes dominate the worst clear cases, a prospective test should separate condition-sensitive feature change from broad color drift using unchanged clear-image controls. Global mean correction alone is not established as a remedy; target-derived analytical components cannot be deployed as restoration.',
        'independent_identity_holdout_native_CCTV_FINAL_or_Zamboanga_claim':False,'model_qualified':False,
        'neural_model_gradient_optimizer_training_VM_calls':0,'consumed_files':consumed,'all_consumed_before_after_exact':True}
    for name,binding in list(consumed.items()):need(pin(ROOT/name)==binding,'Consumed file changed at final photometric readback')
    need(not blocker.attempted,'No prohibited model/framework/worker import')
    with(OUT/'photometric_diagnosis_v1.json').open('x',encoding='utf-8',newline='\n')as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'complete':True,'clear_photographs':8,'diagnostic_panels':len(cells),'visual':str(visual),
        'report':str(OUT/'photometric_diagnosis_v1.json'),'model_qualified':False}))


if __name__=='__main__':main()

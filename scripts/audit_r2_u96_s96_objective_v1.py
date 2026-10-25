"""Saved-only output/provenance audit and descriptive metrics. No model calls."""
import base64
from collections import Counter
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import sys
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/codeformer_r2_u96_s96_objective_v1'
ARMS=('r2','spatial_u','spatial_s')
SHA={'r2':'d4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896',
     'spatial_u':'958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0',
     'spatial_s':'7e06314b4c467f3bbd9bc117a5ff73fd33ab2dd76809ef2c99daff315fd76d71'}


def digest(data):return hashlib.sha256(data).hexdigest()


def pin(p):
    b=p.read_bytes();return {'path':p.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':digest(b)}


def bound(row):
    p=ROOT/row['path'];assert p.resolve().is_relative_to(ROOT) and not p.is_symlink()
    b=p.read_bytes();assert len(b)==row['bytes'] and digest(b)==row['sha256'],row['path']
    return b


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def blob(value):return base64.b64decode(value.split(',',1)[1],validate=True)


def mean(values):return float(np.mean(values,dtype=np.float64))


def summary(rows, label):
    if not rows:return {'label':label,'cases':0}
    metrics=['mae','psnr','ssim','adjacent_RGB_residual_gradient_mae',
             'mean_centered_residual_mae','absolute_luminance_bias','signed_luminance_bias']
    result={'label':label,'cases':len(rows),'reference_photographs':len(set(r['reference_id'] for r in rows)),
            'arms':{},'PNG_pairwise':{}}
    for arm in ARMS:
        result['arms'][arm]={kind:{m:mean([r['paired_metrics'][arm][kind][m] for r in rows
            if r['paired_metrics'][arm][kind][m] is not None]) for m in metrics} for kind in ('RAW','PNG')}
    for challenger,reference in [('spatial_u','r2'),('spatial_s','r2'),('spatial_u','spatial_s')]:
        pair_key=challenger+'_vs_'+reference
        result['PNG_pairwise'][pair_key]={}
        for metric,lower in [('mae',True),('ssim',False),('adjacent_RGB_residual_gradient_mae',True),
                             ('mean_centered_residual_mae',True)]:
            differences=[r['paired_metrics'][challenger]['PNG'][metric]-r['paired_metrics'][reference]['PNG'][metric] for r in rows]
            factor=-1 if lower else 1
            result['PNG_pairwise'][pair_key][metric]={
                'better':sum(d*factor>1e-12 for d in differences),'equal':sum(abs(d)<=1e-12 for d in differences),
                'worse':sum(d*factor < -1e-12 for d in differences),
                'mean_difference_challenger_minus_reference':mean(differences)}
    return result


def main():
    start=time.monotonic();assert not (OUT/'saved_output_audit_v1.json').exists()
    run=json.loads((OUT/'comparison.json').read_bytes());assert run['complete'] and len(run['cases'])==68
    plan=json.loads(bound(run['plan']));assert plan['limits']['new_forwards']==run['new_forwards']==174
    assert run['reused_outputs']==30
    cohort=json.loads(bound(plan['cohort']));assert len(cohort['cases'])==68
    helper=ROOT/'scripts/audit_codeformer_spatial_dev_local_v1_r2.py'
    assert pin(helper)['sha256']=='619a7863031ede9d1dcb78fa0bd116014e514cfc7ee1e5fb3d4c7155bc91fb2b'
    spec=importlib.util.spec_from_file_location('retained_independent_metrics',helper)
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    rows=[];counts=Counter();CSV=[];file_count=0
    for source,saved in zip(cohort['cases'],run['cases']):
        assert source['id']==saved['id'] and set(saved['outputs'])==set(ARMS)
        for key in ('profile','reference_id','cohort'):assert source[key]==saved[key]
        upload=bound(source['upload_source']);native=audit.upload_rgb(upload)
        canvas,support,geometry=audit.prepare_independent(native)
        assert geometry==source['geometry'] and np.array_equal(canvas,audit.rgb(bound(source['input'])))
        assert np.array_equal(support,audit.support_png(bound(source['support'])))
        target=audit.rgb(bound(source['target'])).astype(np.float32)/np.float32(255) if source['target'] else None
        decision=source['input_review'].get('decision',source['input_review'].get('input_review','unknown'))
        row={'id':source['id'],'reference_id':source['reference_id'],'source':source['source'],
             'cohort':source['cohort'],'profile':source['profile'],'input_review':decision,
             'appearance_flag':source.get('appearance_flag',False),'paired_metrics':{},
             'native_PSNR_SSIM':None,'distance_to_R2_not_quality':{},'output_pins':saved['outputs']}
        values={}
        for arm in ARMS:
            pins=saved['outputs'][arm];data={k:bound(v) for k,v in pins.items() if isinstance(v,dict)}
            file_count+=len(data)
            body=json.loads(data['body']);headers={k.lower():v for k,v in json.loads(data['headers']).items()}
            assert int(headers['content-length'])==len(data['body'])
            assert blob(body['bundle'])==data['ZIP'] and set(body['outputs'])=={arm}
            meta=json.loads(data['metadata']);assert body['metadata']==meta
            assert meta['geometry']==geometry and meta['uploaded_bytes_sha256']==digest(upload)
            assert meta['device']=='cpu' and meta['inference_only'] and not meta['main_app_default_changed'] and not meta['model_qualified']
            with zipfile.ZipFile(io.BytesIO(data['ZIP'])) as z:
                assert len(z.namelist())==6 and len(set(z.namelist()))==6 and z.testzip() is None
                assert json.loads(z.read('processing.json'))==meta
                assert z.read(arm+'.png')==data['PNG'] and z.read(arm+'-raw.npy')==data['RAW']
                assert np.array_equal(audit.rgb(z.read('resize-control.png')),canvas)
                assert np.array_equal(audit.rgb(z.read('original-upload.png'),exact_size=None),native)
                assert np.array_equal(audit.support_png(z.read('observed-support.png')),support)
            raw=audit.raw_array(data['RAW']);image=audit.rgb(data['PNG'])
            assert np.array_equal(image,np.floor(raw*np.float32(255)).astype(np.uint8))
            assert np.array_equal(image[~support],canvas[~support])
            assert np.array_equal(raw[~support],(canvas.astype(np.float32)/np.float32(255))[~support])
            declaration=meta['outputs'][arm];provenance=declaration['metadata']
            assert provenance['weights_sha256']==SHA[arm] and provenance['arm']==arm
            assert provenance['fidelity']==1. and provenance['adain'] is True
            assert provenance['display_processing']=='none'
            assert digest(raw.tobytes())==declaration['raw_rgb_sha256']
            assert digest(image.tobytes())==declaration['PNG_RGB8_sha256']
            assert digest(data['PNG'])==declaration['PNG_file_sha256']
            assert blob(body['outputs'][arm])==data['PNG']
            values[arm]={'RAW':raw,'PNG':image.astype(np.float32)/np.float32(255)}
            if target is not None:
                row['paired_metrics'][arm]={kind:audit.metrics(value,target,support) for kind,value in values[arm].items()}
                for kind,metrics in row['paired_metrics'][arm].items():
                    CSV.append({'case':source['id'],'reference':source['reference_id'],'profile':source['profile'],
                        'readability':decision,'appearance_flag':row['appearance_flag'],'arm':arm,'format':kind,
                        **{k:metrics[k] for k in ['mae','psnr','ssim','adjacent_RGB_residual_gradient_mae',
                            'mean_centered_residual_mae','absolute_luminance_bias','signed_luminance_bias']}})
            counts[pins['origin']]+=1
        for arm in ARMS:
            row['distance_to_R2_not_quality'][arm]={kind:float(np.abs((values[arm][kind].astype(np.float64)-
                values['r2'][kind].astype(np.float64))[support]).mean()) for kind in ('RAW','PNG')}
        if target is not None:row['resize_control_metrics']=audit.metrics(canvas.astype(np.float32)/np.float32(255),target,support)
        rows.append(row)
    assert counts=={'prior_saved_exact_reuse':30,'new_local_frozen_inference':174}
    paired=[r for r in rows if r['paired_metrics']]
    summaries=[summary(paired,'all36paired_DEV_four_reference_photographs'),
        summary([r for r in paired if r['profile']!='clear'],'all32degraded'),
        summary([r for r in paired if r['profile']=='clear'],'all4clear_controls'),
        summary([r for r in paired if r['input_review']=='readable'],'input_readable12'),
        summary([r for r in paired if r['input_review']=='readable' and r['profile']!='clear'],'input_readable_degraded9'),
        summary([r for r in paired if r['input_review']=='borderline'],'input_borderline24'),
        summary([r for r in paired if not r['appearance_flag']],'appearance_unflagged27'),
        summary([r for r in paired if r['appearance_flag']],'appearance_flagged9')]
    summaries += [summary([r for r in paired if r['reference_id']==ref],'reference_'+ref) for ref in sorted(set(r['reference_id'] for r in paired))]
    summaries += [summary([r for r in paired if r['profile']==profile],'profile_'+profile) for profile in sorted(set(r['profile'] for r in paired))]
    for metric,lower in [('mae',True),('ssim',False),('adjacent_RGB_residual_gradient_mae',True)]:
        for r in paired:
            order=sorted(ARMS,key=lambda a:r['paired_metrics'][a]['PNG'][metric],reverse=not lower)
            r.setdefault('descriptive_PNG_metric_order',{})[metric]=order
    native_summary={}
    for group in ('native_ChokePoint','native_QMUL-SurvFace'):
        selected=[r for r in rows if r['cohort']==group]
        native_summary[group]={'cases':len(selected),'reference_labels':len(set(r['reference_id'] for r in selected)),
            'aligned_clean_target':False,'PSNR_SSIM':None,'quality_order':None,
            'mean_RGB8_difference_to_R2_observed_support_not_quality':{a:mean([r['distance_to_R2_not_quality'][a]['PNG']*255 for r in selected]) for a in ARMS}}
    report={'complete':True,'audit_pass':True,'auditor':pin(Path(__file__)),'comparison':pin(OUT/'comparison.json'),
        'plan':pin(OUT/'comparison_plan_v1.json'),'retained_independent_metric_helper':pin(helper),
        'counts':{'cases':68,'paired':36,'native_unpaired':32,'new_forwards':174,'reused_outputs':30,
                  'output_file_bindings':file_count,'input_only_refusal_controls_excluded':40},
        'summary':summaries,'native_source_separate':native_summary,'cases':rows,
        'metrics_are_not_identity_or_facial_structure_qualification':True,'only4paired_reference_photographs':True,
        'repeat_DEV_pretraining_overlap_limits_retained':True,'independent_FINAL_review':False,
        'author_saved_only_audit_not_independent_human_review':True,'auditor_neural_HTTP_training_VM_calls':0,
        'model_qualified':False,'main_app_default_changed':False,'seconds':time.monotonic()-start}
    write(OUT/'saved_output_audit_v1.json',report)
    with (OUT/'paired_metrics_v1.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(CSV[0]));writer.writeheader();writer.writerows(CSV)
    make_sheets(cohort,run)
    assert 'torch' not in sys.modules
    print(json.dumps({'audit_pass':True,'paired':36,'native':32,'summaries':len(summaries),'report':pin(OUT/'saved_output_audit_v1.json')}))


def make_sheets(cohort,run):
    visual=OUT/'visuals';visual.mkdir()
    # Coded panels reduce label emphasis; this is root review, not an independent blinded study.
    order=list(ARMS);random.Random(20261021).shuffle(order)
    write(visual/'panel_key.json',{'A':order[0],'B':order[1],'C':order[2],
        'formal_blinded_independent_study':False})
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
    groups=[]
    for group in ('filter_DEV','native_ChokePoint','native_QMUL-SurvFace'):
        group_cases=[(s,r) for s,r in zip(cohort['cases'],run['cases']) if s['cohort']==group]
        for offset in range(0,len(group_cases),4):groups.append((group,group_cases[offset:offset+4]))
    manifest={'complete':True,'review_complete':False,'sheets':[],'panel_key':pin(visual/'panel_key.json')}
    for index,(group,cases) in enumerate(groups,1):
        sheet=Image.new('RGB',(1348,85+len(cases)*294),(245,243,236));draw=ImageDraw.Draw(sheet)
        draw.text((10,8),f'{group} | fixed original inputs; exact256 PNGs; no display processing | sheet{index}',font=font,fill='black')
        draw.text((10,30),'Paired rows have a clean target. Native rows are unpaired; no accuracy score or recovered identity claim.',font=small,fill='black')
        for column,label in enumerate(['Input / resize','Clean target / native note','Model A','Model B','Model C']):
            draw.text((10+column*268,59),label,font=font,fill='black')
        ids=[]
        for row_index,(source,saved) in enumerate(cases):
            y=85+row_index*294;ids.append(source['id'])
            decision=source['input_review'].get('decision',source['input_review'].get('input_review','?'))
            draw.text((10,y),source['id']+' | input:'+decision+(' | appearance flag' if source.get('appearance_flag') else ''),font=small,fill='black')
            y+=23
            with Image.open(io.BytesIO(bound(source['input']))) as image:sheet.paste(image.convert('RGB'),(10,y))
            if source['target']:
                with Image.open(io.BytesIO(bound(source['target']))) as image:sheet.paste(image.convert('RGB'),(278,y))
            else:draw.multiline_text((288,y+75),'Native CCTV: unpaired\nNo clean reference\nReview visible preservation',font=font,fill='black',spacing=7)
            for column,arm in enumerate(order,2):
                with Image.open(io.BytesIO(bound(saved['outputs'][arm]['PNG']))) as image:sheet.paste(image.convert('RGB'),(10+column*268,y))
        path=visual/f'{index:02d}_{group}.png';sheet.save(path)
        manifest['sheets'].append({'file':pin(path),'case_ids':ids,'model_output_cells':len(cases)*3})
    write(visual/'manifest.json',manifest)


if __name__=='__main__':main()

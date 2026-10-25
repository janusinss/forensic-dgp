"""Independent fixed68 saved-output audit. NumPy/Pillow only; no model execution.

Requires a terminal supervisor receipt and fresh peer source review. Complete
returns get all136 delivery reconstructions, paired metrics and17 exact-pixel
sheets. Partial stops retain truthful inventory/counters and no fabricated pass.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import stat
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_preparation'
PLAN = PREP / 'fixed_DEV_plan_v1.json'
PLAN_SHA = 'bb8a60f29630ddb21a2e9cea7e43b30c689b78f6474e6671a28c516e21bf1a6d'
RUN = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_run_v1'
OUT = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_saved_audit_v1'
ARMS = ['incumbent_U96', 'postfusion_LR_u64']
MAX_FILES = 3000
MAX_BYTES = 1024**3
MAX_SECONDS = 300


def need(ok, message):
    if not ok: raise ValueError(message)


def digest(raw): return hashlib.sha256(raw).hexdigest()


def strict_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, 'Duplicate JSON key'); out[key] = value
        return out
    def bad(value): raise ValueError('Nonfinite JSON ' + value)
    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)
    def finite(value):
        if type(value) is float: need(math.isfinite(value), 'Nonfinite JSON exponent')
        elif isinstance(value, dict):
            for child in value.values(): finite(child)
        elif isinstance(value, list):
            for child in value: finite(child)
    finite(result); return result


def ordinary(path, maximum=MAX_BYTES):
    path = Path(path).absolute()
    need(path.is_relative_to(ROOT), 'Workspace input only')
    for parent in path.parents:
        info = parent.lstat()
        need(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode)
             and not getattr(info, 'st_file_attributes', 0) & 1024, 'Ordinary directory required')
    info = path.lstat()
    need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and not stat.S_ISLNK(info.st_mode)
         and not getattr(info, 'st_file_attributes', 0) & 1024 and 0 <= info.st_size <= maximum,
         'Bounded unlinked regular file required')
    return path


def pin(path):
    path = ordinary(path); before = path.stat(); h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): h.update(block)
    after = path.stat()
    need((before.st_size, before.st_mtime_ns, before.st_ino) ==
         (after.st_size, after.st_mtime_ns, after.st_ino), 'Input changed during audit')
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': before.st_size, 'sha256': h.hexdigest()}


def read(path): return strict_json(ordinary(path, 32*1024**2).read_bytes())


def bound(row):
    actual = pin(ROOT / row['path'])
    need(all(actual[key] == row[key] for key in ('path', 'bytes', 'sha256')), 'Exact file pin differs: '+row['path'])
    return ROOT / row['path']


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    return pin(path)


def image(path, mode='RGB', size=(256,256)):
    with Image.open(ordinary(path, 16*1024**2)) as im:
        need(im.mode == mode and (size is None or im.size == size) and getattr(im,'n_frames',1)==1,
             'Exact single image mode/geometry required')
        need(im.width*im.height <= 4096**2, 'Bounded source image geometry')
        return np.asarray(im).copy()


def array(path, shape):
    raw = ordinary(path, 4*1024**2).read_bytes(); stream = io.BytesIO(raw)
    version = np.lib.format.read_magic(stream); need(version in ((1,0),(2,0)), 'Known NPY header')
    reader = np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
    actual, fortran, dtype = reader(stream,max_header_size=4096)
    need(actual==shape and dtype==np.dtype(np.float32) and not fortran and not dtype.hasobject
         and stream.tell()+math.prod(shape)*4==len(raw), 'Exact NPY schema/extent before allocation')
    value = np.frombuffer(raw,dtype=np.float32,offset=stream.tell()).reshape(shape)
    need(np.isfinite(value).all(), 'Finite saved FP32 values required'); return value


def down2(value):
    """Fixed512->256 alignFalse bilinear: horizontal .5 blend, then vertical .5.

    This preserves the CPU kernel's separable FP32 operation order, which is
    distinguishable from a flat four-quarter sum at rounding boundaries.
    """
    need(value.dtype==np.float32 and value.shape==(1,3,512,512), 'NCHW512 FP32 only')
    horizontal=value[:,:,:,0::2]*np.float32(.5)+value[:,:,:,1::2]*np.float32(.5)
    result=horizontal[:,:,0::2,:]*np.float32(.5)+horizontal[:,:,1::2,:]*np.float32(.5)
    return result[0].transpose(1,2,0).copy()


def delivery(native, canvas, observed):
    need(canvas.shape==(256,256,3) and canvas.dtype==np.uint8 and observed.shape==(256,256)
         and observed.dtype==np.bool_, 'Exact canvas/support required')
    normalized=(native+np.float32(1))/np.float32(2)
    unclipped=down2(normalized)
    raw=down2(np.clip(normalized,np.float32(0),np.float32(1)))
    raw[~observed]=canvas.astype(np.float32)[~observed]/np.float32(255)
    return unclipped,raw,np.floor(raw*np.float32(255)).astype(np.uint8)


def prepare(original):
    h,w=original.shape[:2]; side=max(h,w);top,left=(side-h)//2,(side-w)//2
    square=Image.new('RGB',(side,side),(128,128,128));square.paste(Image.fromarray(original),(left,top))
    support=Image.new('L',(side,side),0);support.paste(255,(left,top,left+w,top+h))
    canvas=np.asarray(square.resize((256,256),Image.Resampling.BILINEAR)).copy()
    observed=np.asarray(support.resize((256,256),Image.Resampling.NEAREST)).copy()>0
    yy,xx=np.where(observed);bounds=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
    return canvas,observed,{'native_size':[w,h],'native_center_offset':[left,top],'observed_bounds_256':bounds,
        'output_size':[256,256],'policy':'dgp256-observed-canvas-v1','input_rgb_sha256':digest(original.tobytes()),
        'canvas_rgb_sha256':digest(canvas.tobytes()),'observed_sha256':digest(observed.tobytes()),
        'removal_mask_sha256':digest(np.zeros((256,256),dtype=bool).tobytes())}


def regional(value,target,mask,whole=False):
    residual=value.astype(np.float64)-target.astype(np.float64)
    hmask=mask[:,:-1]&mask[:,1:];vmask=mask[:-1,:]&mask[1:,:]
    horizontal=np.abs(residual[:,1:]-residual[:,:-1])[hmask]
    vertical=np.abs(residual[1:]-residual[:-1])[vmask]
    count=horizontal.size+vertical.size
    adjacent=(float((horizontal.mean()+vertical.mean())/2) if whole else
              float((horizontal.sum()+vertical.sum())/count) if count else None)
    return {'MAE':float(np.abs(residual[mask]).mean()) if mask.any() else None,
            'adjacent_residual_RGB_error':adjacent,'pixels':int(mask.sum()),
            'horizontal_internal_edges':int(hmask.sum()),'vertical_internal_edges':int(vmask.sum())}


def all_regions(value,target,observed):
    central=np.zeros((256,256),bool);central[64:192,64:192]=True
    return {name:regional(value,target,mask,whole=(name=='whole' and observed.all())) for name,mask in
            [('whole',observed),('central128',observed&central),('outside_central128',observed&~central)]}


def ssim(value,target):
    """RGB mean of valid11x11 Gaussian SSIM, sigma1.5, population covariance."""
    value=value.astype(np.float64);target=target.astype(np.float64)
    weights=np.exp(-np.square(np.arange(-5,6,dtype=np.float64))/(2*1.5**2));weights/=weights.sum()
    def blur(x):
        h=sum(weights[i]*x[:,i:i+246,:] for i in range(11))
        return sum(weights[i]*h[i:i+246,:,:] for i in range(11))
    a,b=blur(value),blur(target);aa=blur(value*value)-a*a;bb=blur(target*target)-b*b;ab=blur(value*target)-a*b
    return float((((2*a*b+.01**2)*(2*ab+.03**2))/((a*a+b*b+.01**2)*(aa+bb+.03**2))).mean())


def paired(value,target,observed):
    need(observed.all(), 'Paired PSNR/SSIM require exact fully observed aligned target')
    residual=value.astype(np.float64)-target.astype(np.float64);mse=float(np.square(residual).mean())
    return {'regions':all_regions(value,target,observed),'MSE':mse,
            'PSNR_dB':None if mse==0 else float(-10*math.log10(mse)),
            'PSNR_infinite_exact_match':mse==0,'SSIM_RGB_valid11_gaussian1p5_population':ssim(value,target)}


def ordered_mean(values):
    total=0.;count=0
    for value in values:total+=float(value);count+=1
    need(count>0,'Nonempty scalar mean');return total/count


def classify(status,supervisor,comparison,failure,pending,records):
    counters=None if status is None else status.get('counters')
    attempts=None if counters is None else counters.get('full_forward_attempts')
    returns=None if counters is None else counters.get('full_forward_returns')
    unknown=pending or status is None or bool(status.get('forward_outcome_unknown',True)) or attempts!=returns
    full=bool(comparison and comparison.get('complete')is True and supervisor.get('worker_exit')==0
              and not supervisor.get('timeout') and not failure and not unknown and attempts==returns==136
              and counters.get('successful_saved_outputs')==136 and counters.get('completed_cases')==68 and records==136)
    return {'classification':'COMPLETE_FIXED68_SAVED_RETURN' if full else 'PARTIAL_OR_STOPPED_FIXED_DEV_RETURN',
        'full_saved_pixel_phase_supported':full,'forward_attempts':attempts,'forward_returns':returns,
        'forward_outcome_unknown':unknown,'complete_case_count':None if counters is None else counters.get('completed_cases'),
        'complete_arm_record_count':records,'worker_exit':supervisor.get('worker_exit'),'timeout':supervisor.get('timeout'),
        'failure_present':failure is not None,'pending_status_present':pending,'model_qualified':False}


def audit(review_path):
    started=time.monotonic();review=read(review_path)
    need(review['complete'] and review['source_review_pass'] and pin(Path(__file__)) in review['sources'], 'Fresh independent auditor source review')
    for row in review['sources']:bound(row)
    need(pin(PLAN)['sha256']==PLAN_SHA,'Exact frozen fixed68 plan');plan=read(PLAN)
    need(plan['arms']==ARMS and len(plan['cases'])==68 and plan['case_order']==[r['id']for r in plan['cases']], 'Fixed case/arm order')
    supervisor_path=PREP/'fixed_DEV_supervisor_v1.json'
    need(supervisor_path.exists(),'Actual supervisor terminal receipt required; do not audit live run')
    supervisor=read(supervisor_path);need(supervisor['complete']and supervisor['plan_sha256']==PLAN_SHA,'Terminal supervisor plan')
    need(not OUT.exists(),'New immutable audit namespace required')
    files={};total=0
    for path in sorted(RUN.rglob('*')):
        need(not path.is_symlink()and not getattr(path.lstat(),'st_file_attributes',0)&1024,'No returned link/reparse')
        if path.is_file():
            need(len(files)<MAX_FILES,'Bounded return inventory');row=pin(path);total+=row['bytes'];need(total<=MAX_BYTES,'Bounded return bytes');files[row['path']]=row
    status=read(RUN/'status.json') if(RUN/'status.json').exists()else None
    comparison=read(RUN/'comparison.json')if(RUN/'comparison.json').exists()else None
    failure=read(RUN/'failure.json')if(RUN/'failure.json').exists()else None
    record_paths=list(RUN.glob('cases/*/*/record.json'))
    classification=classify(status,supervisor,comparison,failure,(RUN/'status.pending').exists(),len(record_paths))
    OUT.mkdir();inventory=write(OUT/'saved_return_inventory_v1.json',{'complete':True,'files':files,'bytes':total,
          'classification':classification,'plan':pin(PLAN),'supervisor':pin(supervisor_path),'pending_status_bytes_preserved':(RUN/'status.pending').exists()})
    if not classification['full_saved_pixel_phase_supported']:
        result=write(OUT/'saved_pixel_audit_v1.json',{'complete':True,'pixel_phase_complete':False,
            'independent_saved_pixel_integrity_pass':False,'return_classification':classification,'inventory':inventory,
            'no_future_outputs_metrics_or_visuals_fabricated':True,'partial_stop_evidence_retained':True,
            'model_qualified':False,'neural_optimizer_training_VM_calls':0})
        print(json.dumps({'complete':True,'partial':True,'audit':result}));return 2
    checks=[]
    def ck(name,ok):
        checks.append({'name':name,'pass':bool(ok)})
        need(ok,name)
    ck('terminal result/status/supervisor exact counters',comparison['counters']==status['counters']==supervisor['counters'])
    counters=status['counters'];ck('no optimization and68perarm counters',all(counters[k]==0 for k in['gradient_queries','backward_calls','optimizer_constructions','optimizer_updates'])and all(counters['per_arm'][arm]=={'attempts':68,'returns':68,'saved':68}for arm in ARMS))
    ck('comparison all68 rows exact plan order',[x['id']for x in comparison['cases']]==plan['case_order'])
    ck('successful source limits not qualification',comparison['model_qualified']is False and comparison['app_default_changed']is False and comparison['paired_and_native_evidence_separate'])
    for row in [*plan['sources'],*plan['prerequisites'].values(),plan['protocol'],plan['head_file'],plan['cohort'],plan['data_reuse'],*plan['model'].values()]:bound(row)
    controls=read(RUN/'controls_zero_calls.json');ck('all zero-call controls retained and outside output cases',controls['complete']and controls['neural_calls']==0 and controls['controls']==plan['zero_call_controls']and not({x['id']for x in controls['controls']}&set(plan['case_order'])))
    expected=set();expect=lambda p:expected.add(p.relative_to(ROOT).as_posix())
    for name in ['started.json','status.json','comparison.json','resources.jsonl','controls_zero_calls.json']:expect(RUN/name)
    rows=[];visuals={};byprofile=defaultdict(list)
    for index,(spec,case)in enumerate(zip(plan['cases'],comparison['cases'])):
        need(time.monotonic()-started<MAX_SECONDS,'Finite saved audit time')
        cid=spec['id'];folder=RUN/'cases'/cid;expect(folder/'case.json')
        ck('case receipt exact '+cid,read(folder/'case.json')==case and case['complete']and case['geometry']==spec['geometry']and case['cohort']==spec['cohort']and case['paired_target']==spec['target']and case['native_PSNR_SSIM_identity_metrics']is None)
        for key in ['upload_source','input','support','target','source1024','pair_record']:
            if spec.get(key):bound(spec[key])
        original=image(bound(case['inputs']['original']),size=None);expect(folder/'original.png')
        ck('exact saved common input namespace '+cid,all((ROOT/case['inputs'][key]['path'])==folder/name for key,name in [('original','original.png'),('prepared','prepared.png'),('support','observed_support.png')]))
        upload=image(bound(spec['upload_source']),size=None)
        ck('original savedRGB exact upload '+cid,np.array_equal(original,upload))
        canvas,observed,geometry=prepare(original)
        ck('prepared geometry exact '+cid,geometry==spec['geometry']and digest(canvas.tobytes())==spec['input_pixels']['RGB8_sha256'])
        ck('retained originalderived canvas/support exact '+cid,np.array_equal(canvas,image(bound(spec['input'])))and np.array_equal(observed,image(bound(spec['support']),'L')>0)and np.array_equal(canvas,image(bound(case['inputs']['prepared'])))and np.array_equal(observed,image(bound(case['inputs']['support']),'L')>0))
        for name in ['prepared.png','observed_support.png']:expect(folder/name)
        target=None
        if spec['target']:
            target=image(bound(spec['target']));ck('saved paired clean target exact '+cid,np.array_equal(target,image(bound(case['inputs']['target'])))and case['inputs']['target']['sha256']==spec['target']['sha256']and ROOT/case['inputs']['target']['path']==folder/'target.png');expect(folder/'target.png')
        else:ck('native target absent '+cid,'target'not in case['inputs']and case['paired_target']is None)
        answer={'id':cid,'cohort':spec['cohort'],'profile':spec['profile'],'reference_id':spec['reference_id'],
            'evidence':'paired_synthetic'if target is not None else'native_unpaired','geometry':geometry,'input_only_review':spec['input_review'],'arms':{}}
        rawarms={};pngarms={}
        for arm_index,arm in enumerate(ARMS):
            output=case['outputs'][arm];armfolder=folder/arm;record=read(armfolder/'record.json');expect(armfolder/'record.json')
            ck('arm record case/output exact '+cid+arm,record['complete']and record['case_id']==cid and all(record[key]==value for key,value in output.items()))
            # Per-arm record is written before its saved counter increment.
            n=index*2+arm_index+1;counter=record['counters'];ck('arm durable receipt counter position '+cid+arm,counter['full_forward_attempts']==counter['full_forward_returns']==n and counter['successful_saved_outputs']==n-1 and counter['completed_cases']==index)
            arrays={}
            shapes={'native512':(1,3,512,512),'logits':(1,256,1024),'LQ':(1,256,16,16),'unclipped':(256,256,3),'raw':(256,256,3)}
            schemas={}
            for key,shape in shapes.items():
                path=bound(output[key]);filename={'native512':'native512.npy','logits':'logits.npy','LQ':'LQ.npy','unclipped':'unclipped256.npy','raw':'raw.npy'}[key]
                need(path==armfolder/filename,'Exact per-arm keyed array namespace');expect(path);arrays[key]=array(path,shape)
                schemas[key]={'file':output[key],'shape':list(shape),'dtype':'float32','minimum':float(arrays[key].min()),'maximum':float(arrays[key].max()),'tensor_bytes_sha256':digest(arrays[key].tobytes())}
            unclipped,raw,png=delivery(arrays['native512'],canvas,observed)
            ck('native512 unclipped bilinear256 exact '+cid+arm,np.array_equal(unclipped,arrays['unclipped']))
            ck('native clamp-before-bilinear plus padding RAW exact '+cid+arm,np.array_equal(raw,arrays['raw']))
            pngpath=bound(output['png']);need(pngpath==armfolder/'output.png','Exact keyed PNG namespace')
            outputpng=image(pngpath);expect(armfolder/'output.png')
            ck('RAW float32 floorPNG exact '+cid+arm,np.array_equal(png,outputpng))
            ck('outside support exact prepared pixels '+cid+arm,np.array_equal(raw[~observed],canvas.astype(np.float32)[~observed]/np.float32(255))and np.array_equal(outputpng[~observed],canvas[~observed]))
            provenance=output['provenance'];expected_provenance={'arm':arm,'R2_file_sha256':plan['model']['r2']['sha256'],'U96_file_sha256':plan['model']['u96']['sha256'],'head_file_sha256':plan['head_file']['sha256']if arm_index else None,'head_canonical_sha256':plan['candidate_head_canonical_sha256']if arm_index else None,'w':1.,'detach16':True,'AdaIN':True,'CPU_FP32':True,'display_processing':'none','source_candidate_training_updates':64 if arm_index else None,'model_qualified':False,'same_observed_geometry':True,'outside_support_exact_observed_pixels':True,'forward_sites':['32','64','128','256']}
            ck('exact model and forward policy provenance '+cid+arm,provenance==expected_provenance)
            value={'saved_arrays':schemas,'PNG':output['png'],'native_clipped_components':int(np.count_nonzero((arrays['native512']<-1)|(arrays['native512']>1))),
                'outside_support_RAW_PNG_exact':True,'input_drift':{view:all_regions(v,canvas.astype(np.float64)/255,observed)for view,v in [('raw',raw),('PNG',outputpng.astype(np.float64)/255)]}}
            if target is not None:value['paired']={view:paired(v,target.astype(np.float64)/255,observed)for view,v in [('raw',raw),('PNG',outputpng.astype(np.float64)/255)]}
            else:value['paired_PSNR_SSIM_or_identity']=None
            answer['arms'][arm]=value;rawarms[arm]=raw;pngarms[arm]=outputpng
        answer['candidate_minus_incumbent_output_drift']={view:{'regions':all_regions(values[ARMS[1]],values[ARMS[0]],observed),'maximum_absolute_component_difference':float(np.max(np.abs(values[ARMS[1]].astype(np.float64)-values[ARMS[0]].astype(np.float64))))}for view,values in [('raw',rawarms),('PNG',{k:v.astype(np.float64)/255 for k,v in pngarms.items()})]}
        answer['candidate_vs_incumbent_PNG_changed_pixels']=int(np.any(pngarms[ARMS[0]]!=pngarms[ARMS[1]],axis=2)[observed].sum())
        rows.append(answer);visuals[cid]={'input':canvas,'target':target,'arms':pngarms,'source':spec['source'],'cohort':spec['cohort']}
        if target is not None:byprofile[spec['profile']].append(answer)
    ck('exact complete output namespace',set(files)==expected)
    resources=[strict_json(line)for line in ordinary(RUN/'resources.jsonl',4*1024**2).read_bytes().splitlines()]
    limits=plan['limits'];ck('all274 recorded resource samples within sealed limits',len(resources)==274 and all(x['seconds']<=limits['internal_seconds']and x['worker_RSS_bytes']<=limits['maximum_worker_RSS_bytes']and x['available_RAM_bytes']>=limits['minimum_available_RAM_bytes']and x['free_disk_bytes']>=limits['minimum_free_disk_bytes']and x['output_bytes']<=limits['maximum_output_bytes']for x in resources))
    summaries={}
    for profile,items in byprofile.items():
        need(len(items)==4,'Exactly4paired photos perprofile');entry={'cases':4,'arms':{},'candidate_minus_incumbent':{}}
        for arm in ARMS:
            entry['arms'][arm]={view:{'whole_MAE':ordered_mean(x['arms'][arm]['paired'][view]['regions']['whole']['MAE']for x in items),'whole_adjacent_residual_RGB_error':ordered_mean(x['arms'][arm]['paired'][view]['regions']['whole']['adjacent_residual_RGB_error']for x in items),'central_MAE':ordered_mean(x['arms'][arm]['paired'][view]['regions']['central128']['MAE']for x in items),'central_adjacent_residual_RGB_error':ordered_mean(x['arms'][arm]['paired'][view]['regions']['central128']['adjacent_residual_RGB_error']for x in items),'PSNR_dB_mean':ordered_mean(x['arms'][arm]['paired'][view]['PSNR_dB']for x in items),'SSIM_mean':ordered_mean(x['arms'][arm]['paired'][view]['SSIM_RGB_valid11_gaussian1p5_population']for x in items)}for view in['raw','PNG']}
        entry['candidate_minus_incumbent']={view:{key:entry['arms'][ARMS[1]][view][key]-entry['arms'][ARMS[0]][view][key]for key in entry['arms'][ARMS[0]][view]}for view in['raw','PNG']};summaries[profile]=entry
    ck('paired36/native32 separated with9profiles',len(byprofile)==9 and sum(len(x)for x in byprofile.values())==36 and sum(x['evidence']=='native_unpaired'for x in rows)==32)
    try:font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',14)
    except OSError:font=ImageFont.load_default()
    sheets=[];generated=[]
    for cohort in ['filter_DEV','native_QMUL-SurvFace','native_ChokePoint']:
        ids=[r['id']for r in plan['cases']if r['cohort']==cohort]
        if not ids:continue
        paired_group=cohort=='filter_DEV';columns=['Input','Paired clean','U96','LR u64']if paired_group else['Native prepared input','U96','LR u64']
        for offset in range(0,len(ids),4):
            selected=ids[offset:offset+4];width=16+264*len(columns);height=70+300*len(selected)
            sheet=Image.new('RGB',(width,height),(21,23,27));draw=ImageDraw.Draw(sheet);cells=[]
            draw.text((8,4),cohort+' | fixed DEV | exact256 | '+('paired synthetic'if paired_group else'unpaired: no target/identity claim'),font=font,fill='white')
            for col,title in enumerate(columns):draw.text((8+264*col,30),title,font=font,fill='white')
            filename=f'{cohort}_{offset//4+1:02d}.png'
            for rr,cid in enumerate(selected):
                item=visuals[cid];y=70+300*rr;draw.text((8,y-20),cid,font=font,fill='white')
                arrays=[('input',item['input'],False)]+([('target',item['target'],False)]if paired_group else[])+[(arm,item['arms'][arm],True)for arm in ARMS]
                for col,(label,pixels,isnew)in enumerate(arrays):
                    x=8+264*col;sheet.paste(Image.fromarray(pixels),(x,y));bounds=[x,y,x+256,y+256]
                    path=RUN/'cases'/cid/('prepared.png'if label=='input'else'target.png'if label=='target'else label+'/output.png')
                    cell={'id':cid,'column':columns[col],'source':pin(path),'bounds_xyxy':bounds,'RGB8_sha256':digest(pixels.tobytes()),'generated_output':isnew};cells.append(cell)
                    if isnew:generated.append({'sheet':filename,**cell})
            directory=OUT/'visuals';directory.mkdir(exist_ok=True);path=directory/filename;sheet.save(path)
            with Image.open(path)as im:
                for cell in cells:need(digest(np.asarray(im.crop(cell['bounds_xyxy'])).tobytes())==cell['RGB8_sha256'],'Exact saved256 sheet cell')
            sheets.append({'file':pin(path),'cohort':cohort,'case_ids':selected,'columns':columns,'cells':cells})
    ck('all17 source-separated sheets all68rows and136newPNGs exactlyonce',len(sheets)==17 and len(generated)==136 and len({x['source']['path']for x in generated})==136 and {x['id']for x in generated}==set(plan['case_order']))
    visual=write(OUT/'visual_manifest_v1.json',{'complete':True,'planned_sheets':17,'case_rows':68,'generated_PNG_cells':136,'total_exact256_cells':240,'all_generated_PNGs_exactly_once':True,'no_resampling_or_pixel_overlays':True,'source_groups_separate':True,'root_personal_visual_review_pending':True,'sheets':sheets})
    for row in files.values():ck('returned file remained unchanged '+row['path'],pin(ROOT/row['path'])==row)
    need('torch'not in sys.modules,'No Torch/model imports')
    report={'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),'pixel_phase_complete':True,'independent_saved_pixel_integrity_pass':True,
        'return_classification':classification,'checks':checks,'failures':[],'inventory':inventory,'plan':pin(PLAN),'source_review':pin(review_path),'script':pin(Path(__file__)),
        'case_counts':dict(Counter(x['cohort']for x in rows)),'outputs_verified':136,'all136_native_to_unclipped_RAW_floorPNG_exact':True,'all136_outside_support_exact':True,
        'paired36_metrics_by_profile':summaries,'cases':rows,'paired_metric_definitions':{'regions':'wholly observed pixels; internal edges only; whole fully observed uses historical mean of H/V means; maskedregions use joint internal-edge mean','PSNR':'-10log10 RGB MSE at range1; exact equality is null plus infinite flag','SSIM':'mean RGB SSIM on valid11x11 Gaussian sigma1.5 windows, population covariance, K1.01/K2.03, range1; alignedpaired only'},
        'native_evidence':'32 unpaired cases: observed input/output and cross-arm drift plus exact padding preservation only; no paired PSNR/SSIM, ethnicity or identity recovery claim.',
        'visual_manifest':visual,'root_personal_visual_review_pending':True,'model_qualified':False,'app_default_changed':False,'no_new_quality_threshold_or_pass_claim':True,
        'Torch_neural_optimizer_training_VM_calls':0,'material_limits':['Saved-output integrity and paired numerical changes are separate from visual usefulness; every planned sheet requires root personal review.','Fixed DEV is reused adaptive development evidence, not independentFINAL/user acceptance/native target truth.','All model/frozenhook/RNG behavior is assessed through pinned source/runtime evidence here; no neural call or model state decode is performed.','Input-only readability/exclusion labels stay unchanged; zero-call controls remain excluded; native drift is not restoration accuracy.'],'seconds':time.monotonic()-started}
    result=write(OUT/'saved_pixel_audit_v1.json',report)
    print(json.dumps({'complete':True,'integrity_pass':True,'outputs':136,'sheets':17,'paired_profiles':summaries,'audit':result}));return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-review',type=Path,required=True)
    args=parser.parse_args();return audit(args.source_review)


if __name__=='__main__':raise SystemExit(main())

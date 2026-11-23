"""Independent saved-array/pixel audit; no model, Torch, gradients or VM calls."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,time,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'outputs/completion_codeformer_finetune_v1_r1_return'
RETURN=PARENT/'cctv_completion_codeformer_finetune_v1_r1_results'
OUT=RETURN/'outputs'
ASSETS=ROOT/'outputs/completion_codeformer_finetune_v1_packet/cctv_completion_codeformer_finetune_v1_vm'
EXPECTED='8815ad168ab429bf4f30b8d719d3a6de6d1e2667fc38412c89c9c95e74e5ff3d'
KINDS=('lower_face','eye_band','image_left_lens','cheek_object')
METRICS=('raw_hole_L1','raw_boundary_L1','PNG_hole_L1','PNG_boundary_L1')
def need(ok,label):
    if not ok:raise ValueError(label)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def pin(path):return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}
def read(path):
    def reject(value):raise ValueError('Nonfinite JSON '+value)
    return json.loads(path.read_text(encoding='utf-8-sig'),parse_constant=reject)
def pixels(path,mode='RGB'):
    with Image.open(path) as im:
        need(im.format=='PNG','NonPNG input');return np.array(im.convert(mode),copy=True)
def same(a,b,label):need(a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b),label)
def finite(array,shape):need(array.dtype==np.float32 and array.shape==shape and np.isfinite(array).all(),'Array dtype/shape/finite')
def png(array):return np.rint(np.clip(array[0].transpose(1,2,0),0,1)*np.float32(255)).astype(np.uint8)
def ring(mask):
    pad=np.pad(mask,((3,3),(3,3)),constant_values=True)
    eroded=np.lib.stride_tricks.sliding_window_view(pad,(7,7)).all(axis=(-1,-2))
    return mask & ~eroded
def decode(raw,source,mask):
    decoded=np.clip((raw+np.float32(1))/np.float32(2),0,1)
    top=decoded[:,:,0::2,0::2]+decoded[:,:,0::2,1::2]
    bottom=decoded[:,:,1::2,0::2]+decoded[:,:,1::2,1::2]
    small=(top+bottom)*np.float32(.25)
    result=source.copy();result[:,:,mask]=small[:,:,mask]
    return result
def metric(result,source,mask,delivered,clean):
    b=ring(mask);need(b.any(),'Missing inner boundary')
    difference=np.abs(result-source)[0].transpose(1,2,0).astype(np.float64)
    quant=np.abs(delivered.astype(np.float64)-clean.astype(np.float64))/255
    return {'raw_hole_L1':float(difference[mask].mean()),'raw_boundary_L1':float(difference[b].mean()),'PNG_hole_L1':float(quant[mask].mean()),'PNG_boundary_L1':float(quant[b].mean())}
def aggregate(rows):
    macro={key:sum(row[key] for row in rows)/len(rows) for key in METRICS}
    geometry={kind:{key:sum(r[key] for r in rows if r['mask_kind']==kind)/sum(r['mask_kind']==kind for r in rows) for key in METRICS} for kind in KINDS}
    return {'macro':macro,'geometry':geometry}
def compare(now,old):
    gain=(old['macro']['PNG_hole_L1']-now['macro']['PNG_hole_L1'])/max(old['macro']['PNG_hole_L1'],1e-12)
    nonworse=all(now['macro'][key]<=old['macro'][key]+1e-12 for key in METRICS) and all(now['geometry'][kind][key]<=old['geometry'][kind][key]+1e-12 for kind in KINDS for key in METRICS)
    catastrophic=any(now['macro'][key]>old['macro'][key]*1.1 for key in METRICS) or any(now['geometry'][kind][key]>old['geometry'][kind][key]*1.1 for kind in KINDS for key in METRICS)
    return {'PNG_macro_hole_gain':gain,'all_raw_PNG_hole_boundary_macro_geometry_nonworse':nonworse,'catastrophic_regression_stop':catastrophic,'prospective_numeric_screen_pass':gain>=.01 and nonworse}
def main():
    started=time.monotonic();dest=PARENT/'independent_saved_output_audit.json';need(not dest.exists(),'Preserve existing audit')
    need(sha(RETURN/'protocol.json')==EXPECTED,'Wrong protocol');p=read(RETURN/'protocol.json');need((RETURN/'protocol.json').read_bytes()==(ROOT/'outputs/completion_codeformer_finetune_v1_r1_packet/protocol.json').read_bytes(),'Protocol not sealed bytes')
    manifest=read(RETURN/'export_manifest.json');pins={row['path']:row for row in manifest['files']}
    def evidence(path):
        relative=path.relative_to(RETURN).as_posix();need(relative in pins,'Unmanifested evidence '+relative);row=pins[relative]
        need(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Evidence drift '+relative);return path
    filepins={row['path']:row for row in p['files']};source_pins={}
    def asset(relative,mode='RGB'):
        if relative not in source_pins:
            path=ASSETS/relative;row=filepins[relative];need(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Input/mask asset drift '+relative);source_pins[relative]=pin(path)
        return pixels(ASSETS/relative,mode)
    final=read(evidence(OUT/'results.json'));need(final['cycles_completed']==32 and final['stop_reason'] is None and final['complete'],'Expected32cycle terminal output')
    references={row['id']:row for row in p['references']};fit={r['id']:r for r in p['fit_cases']}
    parity_summary=read(evidence(OUT/'fresh_app_parity/summary.json'));need(parity_summary['complete'] and len(parity_summary['cases'])==4,'Four parity pairs')
    parity_reviews=[]
    for receipt in parity_summary['cases']:
        ident=receipt['id'];case=fit[ident];record=read(evidence(OUT/'fresh_app_parity'/(ident+'.json')));need(record==receipt,'Parity JSON differs fromsummary')
        with np.load(evidence(OUT/'fresh_app_parity'/(ident+'.npz')),allow_pickle=False) as z:
            need(set(z.files)=={'input256','mask256','own_input512','app_input512','own_raw512','app_raw512','own_codes','app_codes','own_result256','app_result256'},'Parity NPZ members')
            arrays={key:z[key].copy() for key in z.files}
        source=arrays['input256'];m=arrays['mask256'];mask=m[0,0].astype(bool);finite(source,(1,3,256,256));finite(m,(1,1,256,256));need(np.isin(m,[0,1]).all(),'Binary parity mask')
        same(png(source),asset(references[case['reference_id']]['clean256']),'Parity actual source');same((m[0,0]*255).astype(np.uint8),asset(case['mask'],'L'),'Parity fixed mask')
        for a,b in [('own_input512','app_input512'),('own_raw512','app_raw512'),('own_codes','app_codes'),('own_result256','app_result256')]:same(arrays[a],arrays[b],'Parity pair differs '+a)
        for side in ('own','app'):
            finite(arrays[side+'_input512'],(1,3,512,512));finite(arrays[side+'_raw512'],(1,3,512,512));finite(arrays[side+'_result256'],(1,3,256,256))
            same(arrays[side+'_result256'][:,:,~mask],source[:,:,~mask],'Parity visible float changed')
            replay=decode(arrays[side+'_raw512'],source,mask);err=float(np.max(np.abs(replay-arrays[side+'_result256'])))
            need(err<=np.finfo(np.float32).eps,'Parity raw-to-result arithmetic exceeds1ulp')
        error=float(np.max(np.abs(arrays['own_result256']-arrays['app_result256'])))
        need(error==record['max_raw256_error']<=1e-6 and record['input512_exact'] and record['input_stride_exact'] and record['own_input_layout']['stride']==record['app_input_layout']['stride'],'Unchanged parity gate failed')
        need(record['max_model_input512_error']==record['model_raw512_max_error']==0 and record['selected_codes_exact'],'Parity zeroerrors differ')
        parity_reviews.append({'id':ident,'source_and_mask_exact':True,'all_four_paired_arrays_exact':True,'raw256_max_error':error,'parity_tolerance':1e-6,'own_and_app_input_stride':record['own_input_layout']['stride'],'raw512_replay_max_error':err})
    evaluations=[];paired_metrics={};summary_metrics={};output_index={};float_exact=0;max_replay_error=0.;raw_to_PNG_differences=0;outsidebytes=0;empty=0;max_raw_metric_deviation=0.;max_png_metric_deviation=0.
    for cycle,kind in [(u,'monitor') for u in (0,8,16,32)]+[(u,'public_development') for u in (0,32)]:
        folder=OUT/('u%04d'%cycle)/kind;summary=read(evidence(folder/'results.json'));cases=p['monitor_cases'] if kind=='monitor' else p['development_cases']
        need(summary['cycle']==cycle and summary['kind']==kind and len(summary['records'])==len(cases)==32,'Evaluation namespace/count')
        rows=[]
        for case,record in zip(cases,summary['records']):
            ident=case['id'];need(record['id']==ident,'Case ordering differs')
            image_path=OUT/record['PNG'];need(image_path==folder/(ident+'.png'),'Output namespace escaped');evidence(image_path);need(sha(image_path)==record['PNG_sha256'],'Output PNG hash')
            expected_source=asset(references[case['reference_id']]['clean256']) if kind=='monitor' else asset(case['input256'])
            mask8=asset(case['mask'],'L');support=mask8==255;need(np.isin(mask8,[0,255]).all() and support.sum()==case['mask_pixels'],'Fixed mask/support differs')
            with np.load(evidence(folder/(ident+'.npz')),allow_pickle=False) as z:
                need(set(z.files)=={'input256','mask256','result256','raw512'},'Output NPZ members');source=z['input256'].copy();m=z['mask256'].copy();result=z['result256'].copy();raw=z['raw512'].copy()
            finite(source,(1,3,256,256));finite(m,(1,1,256,256));finite(result,(1,3,256,256));need(result.min()>=0 and result.max()<=1,'Result range')
            same(png(source),expected_source,'Saved input PNG bytes');same((m[0,0]*255).astype(np.uint8),mask8,'Saved mask exact')
            need(np.max(np.abs(source[0].transpose(1,2,0)-expected_source.astype(np.float32)/255))<=np.finfo(np.float32).eps,'Saved input float not canonical within1ulp')
            delivered=pixels(image_path);same(delivered,png(result),'Saved result-to-rintPNG exact');same(result[:,:,~support],source[:,:,~support],'Visible FP32 source values');same(delivered[~support],expected_source[~support],'Outside PNG original bytes')
            outsidebytes+=int((~support).sum())*3
            if not support.any():
                need(raw.shape==(0,) and raw.dtype==np.float32,'Empty raw must be absent');same(result,source,'Empty source bypass');same(delivered,expected_source,'Empty PNG bypass');empty+=1;replay_error=0.;pngdif=0
            else:
                finite(raw,(1,3,512,512));rebuilt=decode(raw,source,support);replay_error=float(np.max(np.abs(rebuilt-result)));max_replay_error=max(max_replay_error,replay_error)
                need(replay_error<=np.finfo(np.float32).eps,'Raw512-to-result arithmetic exceeds1ulp')
                float_exact+=int(np.array_equal(rebuilt,result));pngdif=int(np.count_nonzero(png(rebuilt)!=delivered));raw_to_PNG_differences+=pngdif
                need(pngdif==0,'Raw512-to-PNG replay mismatch')
            need(record['empty']==(not support.any()) and record['outside_exact'] is True,'Reported support/preservation differs')
            auditrow={'id':ident,'mask_kind':case.get('mask_kind'),'family':case.get('family'),'condition':case.get('condition'),'empty':not bool(support.any()),'PNG':pin(image_path),'raw512_to_result_max_error':replay_error,'raw512_to_PNG_different_channels':pngdif,'saved_result_to_PNG_exact':True,'outside_source_FP32_and_PNG_exact':True}
            if kind=='monitor':
                values=metric(result,source,support,delivered,expected_source);auditrow.update(values);rows.append(auditrow)
                for key in METRICS:
                    deviation=abs(values[key]-record[key])
                    if key.startswith('PNG'):max_png_metric_deviation=max(max_png_metric_deviation,deviation);need(deviation<=1e-14,'PNG paired metric differs')
                    else:max_raw_metric_deviation=max(max_raw_metric_deviation,deviation);need(deviation<=2e-7,'FP32 GPU reduction differs beyond declared tolerance')
            evaluations.append(auditrow);output_index[(cycle,kind,ident)]=image_path
        if kind=='monitor':
            stats=aggregate(rows);paired_metrics[cycle]=stats;summary_metrics[cycle]=summary['metrics']
            for section in ('macro','geometry'):
                a=stats[section];b=summary['metrics'][section]
                if section=='geometry':items=[(a[k][m],b[k][m],m) for k in KINDS for m in METRICS]
                else:items=[(a[m],b[m],m) for m in METRICS]
                for av,bv,key in items:need(abs(av-bv)<=(1e-14 if key.startswith('PNG') else 2e-7),'Summary macro/geometry mismatch')
    need(len(evaluations)==192 and empty==8,'Evaluation/exactemptycount differs')
    independent_screen=compare(paired_metrics[32],paired_metrics[0]);need(independent_screen['prospective_numeric_screen_pass'],'Independent numerical screen fails')
    for key,value in independent_screen.items():
        reported=final['numeric_screen'][key]
        if isinstance(value,bool):need(value==reported,'Final numerical screen bool differs')
        else:need(abs(value-reported)<=1e-14,'Final numeric PNGgain differs')
    screen_rows=[]
    for cycle in (8,16,32):
        current=compare(paired_metrics[cycle],paired_metrics[0]);reported=read(evidence(OUT/('screen_u%04d.json'%cycle)))
        for key,value in current.items():need(value==reported[key] if isinstance(value,bool) else abs(value-reported[key])<1e-14,'Intermediate screen differs')
        screen_rows.append({'cycle':cycle,'independent':current,'VM_receipt':reported})
    empty_receipts=[]
    for cycle in (0,8,16,32):
        receipt=read(evidence(OUT/('empty_controls_u%04d.json'%cycle)));need(receipt['exact_bypasses']==40 and receipt['model_calls']==0 and set(receipt['references'])==set(references),'Fortyemptycontrolreceipt');empty_receipts.append(receipt)
    # All source pixels/masks and evidence stay immutable; recheck bindings.
    for record in source_pins.values():need(sha(Path(record['path']))==record['sha256'],'Input changed during audit')
    need(not any(n=='torch' or n=='pretrained_completion' or n.startswith('third_party.codeformer') for n in sys.modules),'Model runtime imported')
    sheets=PARENT/'visual_sheets';sheets.mkdir(exist_ok=False)
    try:font=ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf',16);smallfont=ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf',13)
    except OSError:font=smallfont=ImageFont.load_default()
    sheet_index=[]
    def overlay(source,mask8):
        a=source.copy();support=mask8==255;a[support]=np.rint(a[support].astype(np.float32)*.4+np.array([230,45,45],np.float32)*.6).astype(np.uint8);return a
    def sheet(name,title,headers,rows):
        width=256*len(headers);canvas=Image.new('RGB',(width,82+286*len(rows)),(241,239,232));draw=ImageDraw.Draw(canvas)
        draw.text((8,7),title,fill=(15,20,24),font=font)
        draw.text((8,30),'Exact saved 256 pixels; no enhancement; all outputs experimental; visual review pending.',fill=(45,45,45),font=smallfont)
        for col,header in enumerate(headers):draw.text((col*256+7,54),header,fill=(15,20,24),font=smallfont)
        slots=[]
        for row,(label,cells,case_id) in enumerate(rows):
            y=82+286*row;draw.text((8,y),label,fill=(15,20,24),font=smallfont)
            for col,array in enumerate(cells):canvas.paste(Image.fromarray(array),(col*256,y+22))
            slots.append({'row':row,'id':case_id,'cells':len(cells)})
        path=sheets/(name+'.png');canvas.save(path)
        # Independently reopen cells for exact placement proof.
        actual=pixels(path)
        for row,(_,cells,_) in enumerate(rows):
            y=82+286*row+22
            for col,array in enumerate(cells):same(actual[y:y+256,col*256:(col+1)*256],array,'Sheet actual pixels differ')
        sheet_index.append({'file':pin(path),'title':title,'rows':slots,'all_displayed_cells_exact':True})
    for number,ref in enumerate(sorted({r['reference_id'] for r in p['monitor_cases']}),1):
        rows=[];clean=asset(references[ref]['clean256'])
        for case in [r for r in p['monitor_cases'] if r['reference_id']==ref]:
            ident=case['id'];m=asset(case['mask'],'L');white=clean.copy();white[m==255]=255
            cells=[clean,white,overlay(clean,m)]+[pixels(output_index[(cycle,'monitor',ident)]) for cycle in (0,8,16,32)]
            rows.append((ident,cells,ident))
        sheet('monitor_%02d_%s'%(number,ref),'Adaptive TRAIN MONITOR | '+ref+' | Known clean target; not independent FINAL',['Known clean target','Masked conditioning example','Fixed removal support (red)','Original inpainting: u0','Trained u8: images only','Trained u16: images only','Terminal trained u32'],rows)
    cases=p['development_cases']
    for offset in range(0,len(cases),4):
        rows=[]
        for case in cases[offset:offset+4]:
            ident=case['id'];source=asset(case['input256']);m=asset(case['mask'],'L');cells=[source,overlay(source,m),pixels(output_index[(0,'public_development',ident)]),pixels(output_index[(32,'public_development',ident)])]
            rows.append((ident+' | '+case['family']+' | '+case['condition'],cells,ident))
        sheet('public_DEV_%02d'%(offset//4+1),'Public DEV photographs | UNPAIRED | No aligned hidden target; native suffix is not CCTV',['Observed input','Reviewed removal support','Original inpainting: u0','Terminal trained u32'],rows)
    need(len(sheet_index)==16,'Expected16 complete sheets')
    index={'format':'completion-r1-complete-visual-index-v1','sheets':sheet_index,'monitor_cases':32,'monitor_snapshots':[0,8,16,32],'public_DEV_cases':32,'public_DEV_snapshots':[0,32],'selection':'Every declared monitor case, all4snapshots; every declared publicDEVcase, baseline+terminal; no ranking/cherrypick','root_visual_review_pending':True}
    with (PARENT/'visual_sheet_index.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(index,f,indent=2);f.write('\n')
    report={'format':'completion-finetune-r1-independent-saved-pixel-audit-v1','complete':True,'independent':True,'audited_UTC':datetime.now(timezone.utc).isoformat(),'protocol':pin(RETURN/'protocol.json'),'auditor':pin(Path(__file__)),'parity':parity_reviews,'parity_pairs':4,'parity_max_output_error':max(r['raw256_max_error'] for r in parity_reviews),'parity_gate_unchanged':1e-6,'evaluated_outputs':192,'raw512_rebuilt_nonempty':184,'raw512_rebuild_bitexact_arrays':float_exact,'raw512_rebuild_max_FP32_error':max_replay_error,'raw512_to_PNG_different_channels':raw_to_PNG_differences,'saved_float256_to_rint_PNG_all_exact':True,'outside_source_FP32_and_PNG_all_exact':True,'outside_source_RGB_bytes_exact':outsidebytes,'empty_saved_output_bypasses_exact':empty,'forty_reference_empty_control_VM_receipts':empty_receipts,'empty_controls_receipt_limitation':'160 no-call reference checks are VM receipts; saved empty DEV arrays give8 independent exact bypass checks. No new model call was made.','paired_MONITOR_metrics_recomputed':paired_metrics,'intermediate_screens':screen_rows,'terminal_independent_numeric_screen':independent_screen,'maximum_metric_receipt_deviations':{'raw_FP32_GPU_reduction':max_raw_metric_deviation,'PNG_float64':max_png_metric_deviation},'reduction_tolerances':{'raw_FP32_GPU_vs_independent_float64_absolute':2e-7,'PNG_float64':1e-14,'parity_gate':1e-6,'raw512_downsample_replay_one_FP32_ulp':float(np.finfo(np.float32).eps),'purpose':'Only cross-runtime arithmetic verification; quality requirements remain1% PNGgain and1e-12 nonworse endpoint rule; independently computed screen uses exact protocol rule.'},'all_outputs':evaluations,'sources':source_pins,'visual_sheets':sheet_index,'scope':'Adaptive originalTRAIN monitor evidence, not independent identities/FINAL. PublicDEV is unpaired original/synthetic-degraded photographs, not native CCTV; no hiddenidentity/PSNR/SSIM claim. Manual reviewed masks only; full automatic and covering-family usefulness remain unqualified.','visual_quality_review_performed_by_auditor':False,'model_qualified':False,'local_execution':{'Torch_imported':False,'model_loads':0,'neural_forwards':0,'gradients':0,'optimizer_steps':0,'VM_calls':0},'audit_seconds':time.monotonic()-started}
    with dest.open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'audit':pin(dest),'outputs':len(evaluations),'gain':independent_screen['PNG_macro_hole_gain'],'outside_bytes':outsidebytes,'max_replay_error':max_replay_error,'sheets':len(sheet_index),'seconds':report['audit_seconds']}))
if __name__=='__main__':main()
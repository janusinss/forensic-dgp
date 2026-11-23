"""Independent actual returned delivered-objective V4 saved-array audit. No Torch/model/VM calls.
Reads only manifest-pinned returned arrays and frozen staged assets; reports
completed and incomplete cohorts separately and never qualifies a partial state.
"""
from __future__ import annotations
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import argparse,ast,hashlib,json,sys,time,zipfile
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL_SHA='8bbb3dd611e95cfbce76e7e6426ffdadb31ce23dadb1dfdcd0246e501213069d'
TRANSPORT_SHA='187be2e99a5439528288ccbdf3d196752b14dd3e8262429046ec9b1958088570'
EXPECTED_RETURN=ROOT/'outputs/completion_codeformer_delivered_objective_v4_return/cctv_completion_codeformer_delivered_objective_v4_results'
EXPECTED_OUT=EXPECTED_RETURN.parent/'independent_saved_output_audit.json'
OBJECTIVE='app_float256_mask_RGB_L1_only_v1'
STAGE=ROOT/'outputs/completion_codeformer_delivered_objective_v4_packet/draft_001/cctv_completion_codeformer_delivered_objective_v4_vm'
BASELINE=ROOT/'outputs/completion_codeformer_finetune_v1_r1_return/cctv_completion_codeformer_finetune_v1_r1_results'
KINDS=('lower_face','eye_band','image_left_lens','cheek_object')
METRICS=('raw_hole_L1','raw_boundary_L1','PNG_hole_L1','PNG_boundary_L1')
RAW_TOL=2e-7
PNG_TOL=1e-14
def need(ok,label):
    if not ok:raise ValueError(label)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def pin(path):return {'path':str(Path(path).resolve()),'bytes':Path(path).stat().st_size,'sha256':sha(path)}
def read(path):
    def reject(value):raise ValueError('Nonfinite JSON '+value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=reject)
def pixels(path,mode='RGB'):
    with Image.open(path) as im:
        need(im.format=='PNG' and im.mode==mode and im.size in ((256,256),(512,512)),'Exact PNG geometry/mode required '+str(path));return np.array(im,copy=True)
def same(a,b,label):need(a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b),label)
def finite(a,shape):need(a.dtype==np.float32 and a.shape==shape and np.isfinite(a).all(),'FP32shape/finite')
def png(a):return np.rint(np.clip(a[0].transpose(1,2,0),0,1)*np.float32(255)).astype(np.uint8)
def ring(m):
    padded=np.pad(m,((3,3),(3,3)),constant_values=True);eroded=np.lib.stride_tricks.sliding_window_view(padded,(7,7)).all(axis=(-1,-2));return m & ~eroded
def decode(raw,source,support):
    a=np.clip((raw+np.float32(1))/np.float32(2),0,1)
    small=((a[:,:,0::2,0::2]+a[:,:,0::2,1::2])+(a[:,:,1::2,0::2]+a[:,:,1::2,1::2]))*np.float32(.25)
    result=source.copy();result[:,:,support]=small[:,:,support];return result
def metric(result,clean_float,support,delivered,target):
    b=ring(support);need(b.any(),'Nonempty inner boundary');raw=np.abs(result-clean_float)[0].transpose(1,2,0).astype(np.float64);quant=np.abs(delivered.astype(np.float64)-target.astype(np.float64))/255
    return {'raw_hole_L1':float(raw[support].mean()),'raw_boundary_L1':float(raw[b].mean()),'PNG_hole_L1':float(quant[support].mean()),'PNG_boundary_L1':float(quant[b].mean())}
def aggregate(rows,condition=False):
    need(rows,'Nonempty aggregate')
    def avg(part):need(part,'Missing group');return {key:sum(r[key] for r in part)/len(part) for key in METRICS}
    result={'macro':avg(rows),'geometry':{kind:avg([r for r in rows if r['mask_kind']==kind]) for kind in KINDS}}
    if condition:result['condition']={name:avg([r for r in rows if r['condition']==name]) for name in sorted({r['condition'] for r in rows})}
    return result
def compare(now,old,condition=False):
    pairs=[('macro',now['macro'],old['macro'])]
    for group in ('geometry','condition') if condition else ('geometry',):
        need(set(now[group])==set(old[group]),'Subgroups differ');pairs.extend((group+'/'+key,now[group][key],old[group][key]) for key in now[group])
    failures=[{'group':label,'metric':key,'candidate':a[key],'baseline':b[key],'delta':a[key]-b[key]} for label,a,b in pairs for key in METRICS if a[key]>b[key]+1e-12]
    gain=(old['macro']['PNG_hole_L1']-now['macro']['PNG_hole_L1'])/max(old['macro']['PNG_hole_L1'],1e-12)
    return {'PNG_macro_hole_gain':gain,'gain_percent':100*gain,'all_raw_PNG_hole_boundary_groups_nonworse':not failures,'nonworse_failures':failures,'catastrophic_regression_stop':any(a[key]>b[key]*1.10 for _,a,b in pairs for key in METRICS),'prospective_numeric_screen_pass':gain>=.01 and not failures,'condition_groups_included':condition,'model_qualified':False}
def close_metrics(actual,reported,label):
    maximum={'raw':0.,'PNG':0.}
    for group in actual:
        sections=[(actual[group],reported[group])] if group=='macro' else [(actual[group][k],reported[group][k]) for k in actual[group]]
        for a,b in sections:
            for key in METRICS:
                err=abs(a[key]-b[key]);typ='PNG' if key.startswith('PNG') else 'raw';maximum[typ]=max(maximum[typ],err);need(err<=(PNG_TOL if typ=='PNG' else RAW_TOL),label+' '+key)
    return maximum
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--returned',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    started=time.monotonic();returned=a.returned.resolve(strict=True);out=returned/'outputs';destination=a.out.resolve();need(not destination.exists(),'Preserve earlier independent audits')
    need(returned==EXPECTED_RETURN.resolve(strict=True) and destination==EXPECTED_OUT.resolve(),'Strict actual V4 return/report roots')
    transport_path=returned.parent/'transport_audit.json';need(sha(transport_path)==TRANSPORT_SHA,'V4 transport admission SHA differs');transport=read(transport_path)
    need(transport['complete'] and transport['protocol_sha256']==PROTOCOL_SHA and transport['all_payload_manifest_hashes_exact'] and transport['closed_namespace_and_safe_paths'],'V4 transport admission incomplete')
    need(sha(returned/'protocol.json')==PROTOCOL_SHA and sha(STAGE/'protocol.json')==PROTOCOL_SHA,'Exact sealedprotocolSHA')
    p=read(returned/'protocol.json');need(p['format']=='completion-codeformer-delivered-objective-v4','Wrong pilot format');manifest=read(returned/'export_manifest.json');need(manifest['protocol_sha256']==PROTOCOL_SHA,'ManifestprotocolSHA')
    pins={r['path']:r for r in manifest['files']};need(len(pins)==len(manifest['files']),'Manifest unique')
    audited_pins={};array_reviews=[];issues=[]
    def evidence(path):
        relative=path.relative_to(returned).as_posix();need(relative in pins,'Unmanifested evidence '+relative);row=pins[relative];need(path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Evidence drift '+relative);audited_pins[relative]=row;return path
    source_pins={};files={row['path']:row for row in p['files']}
    for row in files.values():
        path=STAGE/row['path'];need(path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Frozen full source/data namespace drift '+row['path'])
    worker_path=STAGE/'scripts/cctv_completion_codeformer_delivered_objective_v4_vm.py';helper_path=STAGE/'research/completion_codeformer_delivered_objective_v4/delivered_loss.py'
    need(sha(worker_path)=='cee91da636b5abc452b43f84c330f35b3814192f7b0f2d2e2f3abddb489d62a4' and sha(helper_path)=='737e6674c294c0ea1ea02fec6792abb741897d16a91d198d41377e5e3ba09586','V4 exact objective sources differ')
    worker_source=worker_path.read_text();helper_source=helper_path.read_text();helper_ast=ast.parse(helper_source,feature_version=(3,10))
    need(not any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(helper_ast)),'Pure helper import contract differs')
    for fragment in ["target=clean[case['teacher_key']][0]",'prediction,rgb_loss,loss=delivered_hole_loss(raw,rgb,m,target,interpolate=F.interpolate,microbatches=4)',"clipping=clipping_diagnostic(raw,m512)","'separate_boundary_loss_added':False","'PNG_rounding_or_STE_in_training':False"]:need(fragment in worker_source,'V4 actual loss invocation differs')
    for fragment in ['generated=((raw512+1)/2).clamp(0,1)',"mode='bilinear',align_corners=False",'result[active]=source256[active]*(1-mask256[active])+generated*mask256[active]','hole_RGB_L1=(abs(app_float-clean256)*mask256).sum()/(mask256.sum()*3)','scaled_microbatch_loss=hole_RGB_L1/microbatches','normalized=(raw512.detach()+1)/2']:need(fragment in helper_source,'V4 exact objective/clipping helper differs')
    fit_refs={r['id'] for r in p['references'] if r['role']=='fit'};monitor_refs={r['id'] for r in p['references'] if r['role']=='monitor'}
    need(len(fit_refs)==32 and len(monitor_refs)==8 and not fit_refs&monitor_refs,'Frozen source split differs')
    for row in p['fit_cases']:need(row['reference_id'] in fit_refs and row['clean256'] in files and row['teacher_key']==row['reference_id']+'::'+row['style'],'FIT target/split differs')
    def asset(relative,mode='RGB'):
        path=STAGE/relative
        if relative not in source_pins:
            row=files[relative];need(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Frozenassetdrift '+relative);source_pins[relative]=pin(path)
        return pixels(path,mode)
    def arrays(path,expected):
        evidence(path)
        with zipfile.ZipFile(path) as z:
            members=z.infolist();need(len(members)==len(expected) and len({i.filename for i in members})==len(members),'NPZ membercount')
            need({i.filename for i in members}=={k+'.npy' for k in expected},'ExactNPZschema')
            need(all(not i.is_dir() and not i.flag_bits&1 and i.file_size<=16*1024**2 and i.compress_type in (0,8) and PurePosixPath(i.filename).name==i.filename for i in members),'SafeNPZmembers')
            need(z.testzip() is None,'NPZ CRC')
        with np.load(path,allow_pickle=False) as z:values={key:z[key].copy() for key in z.files}
        need(all(not value.dtype.hasobject and value.dtype.kind in 'fiu' for value in values.values()),'Numeric-only arrays')
        array_reviews.append({'path':path.relative_to(returned).as_posix(),'schema':sorted(values),'CRC_pass':True,'allow_pickle':False,'numeric_only':True});return values
    supervisor=read(evidence(returned/'supervisor.json'));trainer=(evidence(returned/'trainer.log')).read_text(encoding='utf-8',errors='replace')
    training_cycles=sorted(out.glob('training_cycle_*.json'));cycle_rows=[read(evidence(path)) for path in training_cycles]
    need([r['cycle'] for r in cycle_rows]==list(range(1,len(cycle_rows)+1)),'Completedcycleprefix')
    last_cycle=cycle_rows[-1] if cycle_rows else None
    terminal_cycle=len(cycle_rows);need(terminal_cycle in (16,32,64),'Only recorded finite V4 stop cycles admitted by this completed-run audit')
    objective_contract=read(evidence(out/'objective_contract.json'));need(objective_contract['objective']==OBJECTIVE and not objective_contract['boundary_loss_added'] and objective_contract['all_boundary_acceptance_gates_retained'] and objective_contract['microbatch_divisor_once']==4 and not objective_contract['PNG_rounding_or_STE_in_training'] and objective_contract['clipping_metadata_is_detached_diagnostic_only'],'Actual V4 objective receipt differs')
    objective_rows=[]
    fits={r['id']:r for r in p['fit_cases']}
    for row in cycle_rows:
        need(row['microbatch_ids']==p['schedule'][row['cycle']-1] and row['code_losses']==[] and row['code_route_frozen'] and row['objective']==OBJECTIVE,'Actual V4 cycle/schedule/route differs')
        need([v['id'] for v in row['fusion_losses']]==row['microbatch_ids'] and len(row['fusion_losses'])==4,'Four actual FIT losses required')
        need(np.isfinite(row['fusion_preclip_L2']) and row['fusion_gradient_clipping_applied']==(row['fusion_preclip_L2']>1),'Gradient clipping receipt differs')
        for v in row['fusion_losses']:
            case=fits[v['id']];need(v['clean256']==case['clean256'] and v['style']==case['style'] and v['teacher_key']==case['teacher_key'] and v['objective']==OBJECTIVE,'Loss target/source/style differs')
            need(np.isfinite(v['mask_RGB256_L1']) and v['mask_RGB256_L1']>=0 and v['scaled_microbatch_loss']==v['mask_RGB256_L1']/4 and v['microbatch_divisor']==4 and not v['separate_boundary_loss_added'] and not v['PNG_rounding_or_STE_in_training'],'Loss divisor/component receipt differs')
            c=v['clipping'];need(c['raw512_hole_channel_count']==case['mask_pixels']*4*3 and c['clamp_zero_gradient_for_strictly_outside_channels'] and not c['surrogate_gradient_used'],'Clipping semantic/count receipt differs')
            names=('below0_channel_fraction_hole512','above1_channel_fraction_hole512','clipped_channel_fraction_hole512','clipped_pixel_fraction_hole512');need(all(np.isfinite(c[n]) and 0<=c[n]<=1 for n in names) and abs(c[names[0]]+c[names[1]]-c[names[2]])<=1e-7,'Clipping receipt fractions invalid')
            objective_rows.append(v)
    need(not (out/'teacher_labels').exists(),'V4 fusion-only route must not generate teacher labels');teacher_reviews=[]
    parity=read(evidence(out/'fresh_app_parity/summary.json'));need(parity['complete'] and len(parity['cases'])==4,'Fourparitycases');fits={r['id']:r for r in p['fit_cases']};parity_reviews=[]
    for reported in parity['cases']:
        ident=reported['id'];case=fits[ident];record=read(evidence(out/'fresh_app_parity'/(ident+'.json')));need(record==reported,'Paritysummaryrecord')
        z=arrays(out/'fresh_app_parity'/(ident+'.npz'),{'input256','mask256','own_input512','app_input512','own_raw512','app_raw512','own_codes','app_codes','own_result256','app_result256'})
        source=z['input256'];m=z['mask256'];finite(source,(1,3,256,256));finite(m,(1,1,256,256));need(np.isin(m,[0,1]).all(),'Paritybinarymask');support=m[0,0].astype(bool)
        same(png(source),asset(case['input256']),'Parityactualinput');same((m[0,0]*255).astype(np.uint8),asset(case['mask'],'L'),'Parityactualmask')
        same(z['own_input512'],z['app_input512'],'Freshparityinputexact');same(z['own_codes'],z['app_codes'],'Freshparitycodesexact')
        for side in ('own','app'):
            finite(z[side+'_input512'],(1,3,512,512));finite(z[side+'_raw512'],(1,3,512,512));finite(z[side+'_result256'],(1,3,256,256));same(z[side+'_result256'][:,:,~support],source[:,:,~support],'Parityvisiblefloat')
            err=float(np.max(np.abs(decode(z[side+'_raw512'],source,support)-z[side+'_result256'])));need(err<=np.finfo(np.float32).eps,'Paritydecode1ulp')
        error=float(np.max(np.abs(z['own_result256']-z['app_result256'])));rawerr=float(np.max(np.abs(z['own_raw512']-z['app_raw512'])))
        need(error==reported['max_raw256_error']<=1e-6 and rawerr==reported['model_raw512_max_error'] and reported['input512_exact'] and reported['input_stride_exact'] and reported['selected_codes_exact'],'Parityreceiptgates')
        need(reported['own_input_layout']['stride']==reported['app_input_layout']['stride'],'Paritystride');parity_reviews.append({'id':ident,'raw256_max_error':error,'raw512_max_error':rawerr,'input_and_code_exact':True,'outside_float_exact':True,'raw_to_result_max_error':err})
    evaluations=[];cohorts={};groups=[];outside_bytes=0;empty_count=0;max_replay=0.;png_replay_diff=0;raw_deviation=0.;png_deviation=0.;eval_complete=0
    expected_folder_pairs=[]
    for directory in sorted(out.glob('u[0-9][0-9][0-9][0-9]')):
        cycle=int(directory.name[1:]);need(cycle in (0,16,32,64),'Unplannedcyclenamespace')
        for folder in sorted(q for q in directory.iterdir() if q.is_dir()):
            kind=folder.name;need(kind in ('monitor','challenge','public_development'),'Unplannedevalkind')
            cases=p[{'monitor':'monitor_cases','challenge':'challenge_cases','public_development':'development_cases'}[kind]]
            case_map={r['id']:r for r in cases};summary_path=folder/'results.json';summary=read(evidence(summary_path)) if summary_path.exists() else None
            if summary is not None:need(summary['cycle']==cycle and summary['kind']==kind and [r['id'] for r in summary['records']]==[r['id'] for r in cases],'Completesummarycases');eval_complete+=1
            reported={r['id']:r for r in summary['records']} if summary else {};records=[];seen=[]
            filenames={q.stem for q in folder.glob('*.npz')};png_names={q.stem for q in folder.glob('*.png')};need(filenames<=case_map.keys() and png_names<=case_map.keys(),'Unexpectedoutputcase')
            for case in cases:
                ident=case['id'];npz=folder/(ident+'.npz');image_path=folder/(ident+'.png')
                if not image_path.exists() and not npz.exists():continue
                need(image_path.exists(),'ArraywithoutpriorPNG');evidence(image_path);delivered=pixels(image_path);expected_source=asset(case['input256']);mask8=asset(case['mask'],'L');need(np.isin(mask8,[0,255]).all(),'Binaryfixedmask');support=mask8==255
                need(int(support.sum())==case['mask_pixels'],'Actualmaskpixels');same(delivered[~support],expected_source[~support],'ExactoutsidePNGbytes');outside_bytes+=int((~support).sum())*3
                if not npz.exists():
                    issues.append({'scope':'partial_evaluation','cycle':cycle,'kind':kind,'id':ident,'PNG_present':True,'raw_NPZ_present':False,'outside_PNG_exact':True});continue
                z=arrays(npz,{'input256','mask256','result256','raw512'});source=z['input256'];m=z['mask256'];result=z['result256'];raw=z['raw512']
                finite(source,(1,3,256,256));finite(m,(1,1,256,256));finite(result,(1,3,256,256));need(result.min()>=0 and result.max()<=1,'Deliveredfloatrange')
                same(png(source),expected_source,'Actualinputpixels');same((m[0,0]*255).astype(np.uint8),mask8,'Actualmaskarray');need(np.isin(m,[0,1]).all(),'Binarysavedmask')
                need(float(np.max(np.abs(source[0].transpose(1,2,0)-expected_source.astype(np.float32)/np.float32(255))))<=np.finfo(np.float32).eps,'Inputfloatcanonical1ulp')
                same(result[:,:,~support],source[:,:,~support],'Exactoutsidefloat');same(delivered,png(result),'ExactresultrintPNG')
                replay_error=0.;pngdiff=0
                if not support.any():
                    need(raw.dtype==np.float32 and raw.shape==(0,),'Emptyrawabsent');same(result,source,'Emptyfloatbypass');same(delivered,expected_source,'EmptyPNGbytebypass');empty_count+=1
                else:
                    finite(raw,(1,3,512,512));replay=decode(raw,source,support);replay_error=float(np.max(np.abs(replay-result)));need(replay_error<=np.finfo(np.float32).eps,'Raw512resultdecode1ulp');max_replay=max(max_replay,replay_error);pngdiff=int(np.count_nonzero(png(replay)!=delivered));png_replay_diff+=pngdiff;need(pngdiff==0,'Raw512toPNGexact')
                clipping_recomputed=None
                if support.any():
                    normalized=(raw+np.float32(1))/np.float32(2);m512=np.repeat(np.repeat(support,2,0),2,1);sat=((normalized<0)|(normalized>1))[0].transpose(1,2,0)
                    clipping_recomputed={'clipped_channel_fraction_hole512':float(sat[m512].mean()),'clipped_pixel_fraction_hole512':float(sat[m512].any(-1).mean())}
                record=reported.get(ident)
                if record is not None:need(record['PNG_sha256']==sha(image_path) and record['outside_exact'] and record['empty']==(not support.any()),'Outputreceipt')
                auditrow={'cycle':cycle,'kind':kind,'id':ident,'reference_id':case.get('reference_id'),'mask_kind':case.get('mask_kind'),'condition':case.get('condition'),'family':case.get('family'),'style':case.get('style'),'empty':not bool(support.any()),'NPZ':pin(npz),'PNG':pin(image_path),'outside_float_and_PNG_exact':True,'result_to_PNG_exact':True,'raw512_replay_max_error':replay_error,'raw512_replay_PNG_different_channels':pngdiff,'clipping_saved_output_recomputed':clipping_recomputed}
                if kind!='public_development' and support.any():
                    target=asset(case['clean256']);target_float=target.astype(np.float32).transpose(2,0,1)[None]/np.float32(255)
                    if np.array_equal(expected_source,target):target_float=source
                    values=metric(result,target_float,support,delivered,target);auditrow.update(values);auditrow['recomputed_delivered_hole_objective_unscaled']=values['raw_hole_L1'];auditrow['recomputed_once_divided_microbatch_equivalent']=values['raw_hole_L1']/4
                    if record is not None:
                        for key in METRICS:
                            delta=abs(values[key]-record[key]);need(delta<=(PNG_TOL if key.startswith('PNG') else RAW_TOL),'Pairedmetric '+key)
                            if key.startswith('PNG'):png_deviation=max(png_deviation,delta)
                            else:raw_deviation=max(raw_deviation,delta)
                    if case.get('style')=='GRAY':
                        spread=float(np.ptp(result[0].transpose(1,2,0),axis=-1)[support].mean());quantspread=float(np.ptp(delivered.astype(np.int16),axis=-1)[support].mean());auditrow.update(GRAY_raw_mean_channel_spread=spread,GRAY_PNG_mean_channel_spread=quantspread)
                        if record is not None:need(abs(spread-record['GRAY_raw_mean_channel_spread'])<=2e-7 and quantspread==record['GRAY_PNG_mean_channel_spread'] and record['GRAY_chroma_diagnostic_not_selection_gate'],'GRAYdiagnosticrecompute')
                    records.append(auditrow)
                evaluations.append(auditrow);seen.append(ident)
            complete=len(seen)==len(cases) and set(seen)==set(case_map)
            if summary is not None:need(complete,'Completesummarywithmissingarrays')
            group={'cycle':cycle,'kind':kind,'expected_cases':len(cases),'saved_raw_PNG_cases':len(seen),'complete_case_set':complete,'summary_present':summary is not None,'no_missing_output_synthesized':True}
            if kind!='public_development' and complete:
                stats=aggregate(records,condition=kind=='challenge');cohorts[(cycle,kind)]=stats;group['independent_metrics']=stats
                if summary:group['summary_metric_max_deviation']=close_metrics(stats,summary['metrics'],'Pairedaggregates')
            groups.append(group);expected_folder_pairs.append((cycle,kind))
    print(json.dumps({'audited_evaluation_outputs':len(evaluations),'teacher_sets_generated':len(teacher_reviews),'paired_complete_cohorts':len(cohorts),'outside_bytes_exact':outside_bytes}),flush=True)
    # Recompute each retained clear screen on same-L4 frozen R1 baseline.
    screens=[]
    for cycle in (16,32,64):
        key=(cycle,'monitor');baseline_key=(0,'monitor')
        if key in cohorts and baseline_key in cohorts:
            independent=compare(cohorts[key],cohorts[baseline_key]);reported_path=out/('screen_u%04d.json'%cycle)
            reported=read(evidence(reported_path)) if reported_path.exists() else None
            if reported:
                for field in ('prospective_numeric_screen_pass','all_raw_PNG_hole_boundary_groups_nonworse','catastrophic_regression_stop'):
                    need(independent[field]==reported[field],'Screengateboolean '+field)
                need(abs(independent['PNG_macro_hole_gain']-reported['PNG_macro_hole_gain'])<=PNG_TOL,'Screengain')
            screens.append({'cycle':cycle,'independent':independent,'VM_screen_receipt_present':reported is not None})
    clear=compare(cohorts[(terminal_cycle,'monitor')],cohorts[(0,'monitor')]) if (terminal_cycle,'monitor') in cohorts and (0,'monitor') in cohorts else None
    challenge=compare(cohorts[(terminal_cycle,'challenge')],cohorts[(0,'challenge')],condition=True) if (terminal_cycle,'challenge') in cohorts and (0,'challenge') in cohorts else None
    # Permanent original baseline is independently verified saved R1-u0 data, not NN.
    old_audit=ROOT/'outputs/completion_codeformer_finetune_v1_r1_return/independent_saved_output_audit.json'
    need(sha(old_audit)=='ab4bd40838e33bce240d28265ca727c4ac52e96e0b78ff8ad5abc67854b5b65e','Priororiginalsavedpixeladmission')
    old_manifest={r['path']:r for r in read(BASELINE/'export_manifest.json')['files']};old_rows=[];original_comparisons=[]
    for case in p['monitor_cases']:
        ident=case['id'];folder=BASELINE/'outputs/u0000/monitor';npz=folder/(ident+'.npz');image_path=folder/(ident+'.png')
        for path in (npz,image_path):
            row=old_manifest[path.relative_to(BASELINE).as_posix()];need(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Priororiginaloutputdrift')
        with zipfile.ZipFile(npz) as z:need(z.testzip() is None,'PriorbaselineCRC')
        with np.load(npz,allow_pickle=False) as z:source=z['input256'].copy();m=z['mask256'].copy();result=z['result256'].copy();raw=z['raw512'].copy()
        finite(source,(1,3,256,256));finite(m,(1,1,256,256));finite(result,(1,3,256,256));finite(raw,(1,3,512,512));support=m[0,0].astype(bool);target=asset(case['clean256']);delivered=pixels(image_path);same(delivered,png(result),'PriororiginalresultPNG');same(png(source),target,'Priororiginalcleansource');same((m[0,0]*255).astype(np.uint8),asset(case['mask'],'L'),'Priororiginalfixedmask')
        old_rows.append({'id':ident,'mask_kind':case['mask_kind'],**metric(result,source,support,delivered,target)})
        # Model-state identity is audited separately; delivery parity is diagnostic.
        retained_R1=BASELINE/'outputs/u0032/monitor'/(ident+'.png');reference_row=old_manifest[retained_R1.relative_to(BASELINE).as_posix()];need(sha(retained_R1)==reference_row['sha256'],'R1retainedbaselinePNGdrift')
        current_R1=out/'u0000/monitor'/(ident+'.png');original_comparisons.append({'id':ident,'fresh_R1_vs_retained_R1_PNG_different_channels':int(np.count_nonzero(pixels(current_R1)!=pixels(retained_R1)))})
    original=aggregate(old_rows);close_metrics(original,read(STAGE/p['original_monitor_summary'])['metrics'],'Permanentoriginalbaseline')
    cumulative=compare(cohorts[(terminal_cycle,'monitor')],original) if (terminal_cycle,'monitor') in cohorts else None
    # Independent historical R1 compatibility: all48 compact arrays retain exactly
    # the prior audited source arrays; actual new u0 results and PNGs must agree.
    historical=p['historical_control'];compatibility=[]
    compat_report=read(evidence(out/'historical_R1_compatibility.json'))
    need(compat_report['complete'] and compat_report['pass'] and compat_report['all48_before_update'] and compat_report['updates_before']==0 and len(compat_report['cases'])==48,'All48 pre-update compatibility receipt')
    compat_reported={(r['kind'],r['id']):r for r in compat_report['cases']}
    V2=ROOT/'outputs/completion_codeformer_coverage_v2_return/cctv_completion_codeformer_coverage_v2_results'
    V2_admission=ROOT/'outputs/completion_codeformer_coverage_v2_return/independent_saved_output_audit.json'
    need(sha(V2_admission)=='6ac24d0ac4ccbe4d54201e0c907a132b753c2ee84be35ea8adddf6cfde195fbb','Historical joint V2 saved-output admission differs')
    V2_manifest={r['path']:r for r in read(V2/'export_manifest.json')['files']}
    def historical_npz(path):
        row=V2_manifest[path.relative_to(V2).as_posix()]
        need(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Historical V2 raw drift')
        with zipfile.ZipFile(path) as zip_:need(zip_.testzip() is None,'Historical V2 CRC failed')
        with np.load(path,allow_pickle=False) as archive:return {key:archive[key].copy() for key in archive.files}
    for kind in ('monitor','challenge'):
        for row in historical['initial_rows'][kind]:
            ident=row['id'];compact=STAGE/row['result256'];old_image=STAGE/row['PNG']
            with zipfile.ZipFile(compact) as z:need(z.testzip() is None and set(z.namelist())=={'result256.npy'},'Compact historical NPZschema/CRC')
            with np.load(compact,allow_pickle=False) as z:prior=z['result256'].copy()
            finite(prior,(1,3,256,256));old_source=V2/'outputs/u0000'/kind/(ident+'.npz');previous=historical_npz(old_source)
            same(prior,previous['result256'],'Compact historical R1 source arrays bitexact')
            actual_npz=out/'u0000'/kind/(ident+'.npz')
            with np.load(actual_npz,allow_pickle=False) as z:actual=z['result256'].copy()
            raw_error=float(np.max(np.abs(actual-prior)));PNG_exact=sha(out/'u0000'/kind/(ident+'.png'))==sha(old_image)
            need(raw_error<=1e-6 and PNG_exact,'Actual historical R1 preupdate compatibility')
            vm=compat_reported[(kind,ident)];need(vm['max_raw256_error']==raw_error and vm['PNG_bytes_exact']==PNG_exact,'Compatibility receipt differs')
            need(vm['historical_result256_sha256']==sha(compact) and vm['historical_PNG_sha256']==sha(old_image),'Compatibility source pin differs')
            compatibility.append({'kind':kind,'id':ident,'raw256_max_error':raw_error,'PNG_exact':PNG_exact,'compact_source_exact':True})
    need(len(compatibility)==48,'Historical initial cohort count differs')
    # Recompute the joint-arm u64 endpoints from its saved numeric arrays and
    # check that the supplied historical PNG controls are the exact old bytes.
    joint_cohorts={};joint_comparisons={}
    for kind,key in [('monitor','monitor_cases'),('challenge','challenge_cases')]:
        rows=[];joint_map={r['id']:r for r in historical['joint_rows'][kind]}
        for case in p[key]:
            ident=case['id'];prior=historical_npz(V2/'outputs/u0064'/kind/(ident+'.npz'));support=prior['mask256'][0,0].astype(bool)
            target=asset(case['clean256']);priorPNG=pixels(V2/'outputs/u0064'/kind/(ident+'.png'))
            same(priorPNG,pixels(STAGE/joint_map[ident]['PNG']),'Historical joint packaged PNG bytes')
            same(priorPNG,png(prior['result256']),'Historical joint numeric result toPNG')
            target_float=target.astype(np.float32).transpose(2,0,1)[None]/np.float32(255)
            if np.array_equal(asset(case['input256']),target):target_float=prior['input256']
            rows.append({'id':ident,'mask_kind':case['mask_kind'],'condition':case['condition'],**metric(prior['result256'],target_float,support,priorPNG,target)})
        joint_cohorts[kind]=aggregate(rows,condition=kind=='challenge')
        close_metrics(joint_cohorts[kind],read(STAGE/historical['joint_terminal'][kind])['metrics'],'Historical joint summary independently replayed')
        if (terminal_cycle,kind) in cohorts:joint_comparisons[kind]=compare(cohorts[(terminal_cycle,kind)],joint_cohorts[kind],condition=kind=='challenge')
    final=read(evidence(out/'results.json')) if (out/'results.json').exists() else None
    if final:
        need(final['complete'] and final['protocol_sha256']==PROTOCOL_SHA and final['teacher_neural_calls']==0,'Final results scope differs')
        need(final['cycles_completed_newphase']==terminal_cycle and final['objective']==OBJECTIVE and not final['boundary_loss_added'] and not final['PNG_rounding_or_STE_in_training'],'V4 final loss scope differs')
        expected_counts={'forward_slots':8+256+128+32+56,'backward_calls':256,'code_optimizer_steps':0,'fusion_optimizer_steps':64,'empty_bypasses':168} if terminal_cycle==64 else last_cycle['counts']
        if terminal_cycle==64:need(final['counts_newphase']==expected_counts and final['actual_whole_model_forward_calls']==480,'Actual V4 completed counts differ')
        expected_pairs={(0,'monitor'),(0,'challenge'),(0,'public_development'),(terminal_cycle,'challenge'),(terminal_cycle,'public_development')}|{(v,'monitor') for v in (16,32,64) if v<=terminal_cycle}
        need(set(expected_folder_pairs)==expected_pairs and all(g['complete_case_set'] for g in groups),'Actual evaluation cycle/cohort coverage differs')
        need(len(evaluations)==(224 if terminal_cycle==64 else 160 if terminal_cycle==16 else 192),'Actual finite output count differs')
        vm_screen=final['numeric_screen']
        for kind,actual in [('clear_incremental_vs_R1',clear),('challenge_incremental_vs_R1',challenge),('clear_cumulative_vs_original',cumulative)]:
            if actual is not None:
                vm=vm_screen[kind]
                need(actual['prospective_numeric_screen_pass']==vm['prospective_numeric_screen_pass'] and actual['all_raw_PNG_hole_boundary_groups_nonworse']==vm['all_raw_PNG_hole_boundary_groups_nonworse'] and abs(actual['PNG_macro_hole_gain']-vm['PNG_macro_hole_gain'])<=PNG_TOL,'Final quality gate receipt differs '+kind)
        for kind,actual in joint_comparisons.items():
            vm=vm_screen['historical_joint_V2_diagnostic_only'][kind]
            need(abs(actual['PNG_macro_hole_gain']-vm['PNG_macro_hole_gain'])<=PNG_TOL and actual['all_raw_PNG_hole_boundary_groups_nonworse']==vm['all_raw_PNG_hole_boundary_groups_nonworse'],'Historical joint compare receipt differs')
        need(vm_screen['cohorts_never_pooled'] and vm_screen['prospective_numeric_screen_pass']==bool(clear and challenge and clear['prospective_numeric_screen_pass'] and challenge['prospective_numeric_screen_pass']),'Final overall screen differs')
    print(json.dumps({'early_authoritative':True,'clear_gain_percent':None if clear is None else clear['gain_percent'],'clear_nonworse':None if clear is None else clear['all_raw_PNG_hole_boundary_groups_nonworse'],'challenge_gain_percent':None if challenge is None else challenge['gain_percent'],'challenge_nonworse':None if challenge is None else challenge['all_raw_PNG_hole_boundary_groups_nonworse'],'outside_pixels_exact':True,'model_qualified':False}),flush=True)
    control_receipts=[]
    for cycle in (0,16,32,64):
        path=out/('empty_controls_u%04d.json'%cycle)
        if path.exists():
            row=read(evidence(path));need(row['exact_bypasses']==40 and row['model_calls']==0 and set(row['references'])=={r['id'] for r in p['references']},'Emptycontrolreceipt');control_receipts.append(row)
    all_npz={path.relative_to(returned).as_posix() for path in out.rglob('*.npz')};audited_npz={r['path'] for r in array_reviews};need(all_npz==audited_npz,'EveryretainedNPZaudited')
    all_png={path.relative_to(returned).as_posix() for path in out.rglob('*.png')};audited_png={Path(r['PNG']['path']).relative_to(returned).as_posix() for r in evaluations};audited_png|={r['cycle'] and '' for r in []};orphan_png={r['id'] for r in issues if r.get('PNG_present')};need(len(all_png)==len(audited_png)+len(orphan_png),'EveryretainedPNGscope')
    for row in source_pins.values():need(sha(Path(row['path']))==row['sha256'],'Sourcechanged')
    for relative,row in audited_pins.items():need((returned/relative).stat().st_size==row['bytes'] and sha(returned/relative)==row['sha256'],'Evidencechangedduringaudit')
    need(not any(n=='torch' or n.startswith('torch.') or n.startswith('third_party.codeformer') for n in sys.modules),'NoTorch/modelruntime')
    terminal=out/'terminal_training_state.pth';partial=out/'terminal_training_state.pth.partial';final_path=out/'results.json'
    completed_case_sets={f'{row["cycle"]}:{row["kind"]}':row['complete_case_set'] for row in groups}
    last_logged=None
    for line in trainer.splitlines():
        try:
            row=json.loads(line)
            if isinstance(row,dict) and 'cycle' in row and 'counts' in row:last_logged=row
        except (ValueError,TypeError):pass
    report={'format':'completion-codeformer-delivered-objective-v4-independent-saved-output-audit-v1','transport_admission':pin(transport_path),'objective_contract_audit':{'complete':True,'source_worker':pin(worker_path),'source_helper':pin(helper_path),'actual_objective_receipt':objective_contract,'FIT_loss_component_receipts_verified':len(objective_rows),'loss_divisor_exact_once':4,'additive_boundary_loss':False,'PNG_rounding_or_STE':False,'clipping_detached_semantics':True,'actual_FIT_raw_not_saved':'FIT loss and clipping numerical values cannot be independently replayed without FIT raw arrays. Source graph and all receipt arithmetic/schema are checked; saved evaluation objective values are independently recomputed.','first_last_loss_not_a_matched_trajectory':True,'loss_receipts':objective_rows},'actual_terminal_cycle':terminal_cycle,'complete':True,'independent':True,'audited_UTC':datetime.now(timezone.utc).isoformat(),'protocol':pin(returned/'protocol.json'),'auditor':pin(Path(__file__)),'supervisor':supervisor,'operational_status':{'worker_exit':supervisor['worker_exit'],'timeout':supervisor['timeout'],'actual_child_exit':supervisor['actual_child_exit'],'supervisor_seconds':supervisor['seconds'],'reported_complete_cycle_receipts':len(cycle_rows),'last_cycle_receipt':last_cycle,'last_logged_cycle':last_logged,'committed_terminal_present':terminal.exists(),'terminal_partial':pin(partial) if partial.exists() else None,'final_results_json_present':final_path.exists(),'training_and_evaluation_receipts_complete':supervisor['worker_exit'] in (0,2), 'numeric_qualification_separate':True,'partial_file_is_not_qualified_or_deployable_checkpoint':True,'checkpoint_recoverability_audit_owned_separately':True},'parity_pairs':parity_reviews,'parity_max_output_error':max(r['raw256_max_error'] for r in parity_reviews),'teacher_labels':teacher_reviews,'teacher_label_scope':'No teacher arrays or neural teacher calls permitted in delivered-objective V4; style-matched target PNGs are frozen source assets, not new labels.','evaluated_raw_PNG_outputs':len(evaluations),'complete_evaluation_cohorts':eval_complete,'evaluation_cohorts':groups,'all_retained_NPZ_count':len(array_reviews),'no_teacher_NPZ_generated':True,'all_retained_NPZ_CRC_numeric_schema_pass':True,'all_retained_PNG_count':len(all_png),'saved_float256_to_PNG_all_exact':True,'raw512_to_result_max_FP32_error':max_replay,'raw512_to_PNG_different_channels':png_replay_diff,'outside_source_FP32_and_PNG_all_exact':True,'outside_source_RGB_bytes_exact':outside_bytes,'saved_empty_output_bypasses_exact':empty_count,'empty_control_receipts':control_receipts,'empty_control_scope':'160 source-reference no-call checks are VM receipts; eight saved public empty arrays independently replayed. No model was called by auditor.','maximum_metric_receipt_deviations':{'raw_FP32_GPU_vs_float64':raw_deviation,'PNG_float64':png_deviation},'cross_runtime_metric_verification_tolerances':{'raw_absolute':RAW_TOL,'PNG_absolute':PNG_TOL,'raw512_decode_FP32_ulp':float(np.finfo(np.float32).eps),'quality_thresholds_unchanged':True,'prospective_nonworse_threshold':1e-12,'prospective_gain_minimum':.01},'clear_screens_recomputed':screens,'clear_incremental_vs_R1':clear,'challenge_incremental_vs_R1':challenge,'permanent_original_monitor_metrics_recomputed':original,'clear_cumulative_vs_original':cumulative,'all_numeric_screen_endpoints_pass_if_available':bool(clear and challenge and clear['prospective_numeric_screen_pass'] and challenge['prospective_numeric_screen_pass']),'cohorts_never_pooled':True,'fresh_R1_vs_retained_R1_PNG_comparison':original_comparisons,'historical_R1_48case_admission':compatibility,'historical_joint_V2_metrics_recomputed':joint_cohorts,'candidate_vs_historical_joint_V2':joint_comparisons,'all_outputs':evaluations,'source_assets':source_pins,'array_reviews':array_reviews,'partial_evidence_issues':issues,'scope':'Original adaptive paired TRAIN FIT/MONITOR and synthetic TRAIN challenges only; public DEV photographs remain unpaired and do not establish native CCTV, hidden identity or full covering-family utility. Reviewed assisted masks only. No FINAL identities/pixels or new reviewer ratings used.','visual_quality_review_performed_by_auditor':False,'model_qualified':False,'automatic_app_promotion':False,'new_model_packaged':False,'execution':{'Torch_imported':False,'models':0,'neural_calls':0,'gradients':0,'optimizer_steps':0,'VM_calls':0},'seconds':time.monotonic()-started}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'complete':True,'audit':pin(destination),'evaluated_outputs':len(evaluations),'NPZ':len(array_reviews),'outside_bytes':outside_bytes,'clear_gain_percent':clear['gain_percent'] if clear else None,'clear_pass':clear['prospective_numeric_screen_pass'] if clear else None,'challenge_gain_percent':challenge['gain_percent'] if challenge else None,'challenge_pass':challenge['prospective_numeric_screen_pass'] if challenge else None,'model_qualified':False,'seconds':report['seconds']}),flush=True)
if __name__=='__main__':main()

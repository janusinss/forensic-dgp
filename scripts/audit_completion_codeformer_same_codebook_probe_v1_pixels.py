"""Independent actual SAME-codebook probe saved-pixel/index audit, with no Torch/model/VM calls.
All C/D results use unavailable clean-context indices and are diagnostic only.
"""
from __future__ import annotations
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import argparse,hashlib,json,sys,time,zipfile
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
def long(path):
    raw=str(Path(path).absolute());prefix=chr(92)*2+'?'+chr(92)
    return Path(raw if raw.startswith(prefix) else prefix+raw)
PROTOCOL_SHA='51782b57fca9ad9cf7161b0f4451a65e545e3eb2f8f58788fa40b64aec48d1ec'
STAGE=long(ROOT/'outputs/completion_codeformer_same_codebook_probe_v1_packet/draft_001/cctv_completion_codeformer_same_codebook_probe_v1_vm')
RETURN=long(ROOT/'outputs/completion_codeformer_same_codebook_probe_v1_return/cctv_completion_codeformer_same_codebook_probe_v1_results')
KINDS=('lower_face','eye_band','image_left_lens','cheek_object')
METRICS=('raw_hole_L1','raw_boundary_L1','PNG_hole_L1','PNG_boundary_L1')
SCHEMA={'input256','mask256','result256','raw512','predicted_codes','used_codes','arm_code','oracle_unavailable_clean_information'}
PARITY_SCHEMA={'input256','mask256','own_input512','app_input512','own_raw512','app_raw512','own_codes','app_codes','own_result256','app_result256'}
RAW_TOL=2e-7
PNG_TOL=1e-14
CHECKS=0
def need(ok,label):
    global CHECKS
    CHECKS+=1
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
        need(im.format=='PNG' and im.mode==mode and im.size==(256,256),'Exact PNG geometry/mode required '+str(path));return np.array(im,copy=True)
def same(a,b,label):need(a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b),label)
def finite(a,shape):need(a.dtype==np.float32 and a.shape==shape and np.isfinite(a).all(),'Finite FP32 geometry')
def codes(a):need(a.dtype==np.int64 and a.shape==(1,256) and a.min()>=0 and a.max()<512,'Exactly256 finite in-codebook int64 indices')
def png(a):return np.rint(np.clip(a[0].transpose(1,2,0),0,1)*np.float32(255)).astype(np.uint8)
def ring(m):
    padded=np.pad(m,((3,3),(3,3)),constant_values=True);eroded=np.lib.stride_tricks.sliding_window_view(padded,(7,7)).all(axis=(-1,-2));return m & ~eroded
def decode(raw,source,support):
    a=np.clip((raw+np.float32(1))/np.float32(2),0,1)
    small=((a[:,:,0::2,0::2]+a[:,:,0::2,1::2])+(a[:,:,1::2,0::2]+a[:,:,1::2,1::2]))*np.float32(.25)
    result=source.copy();result[:,:,support]=small[:,:,support];return result

def metric(result,target_float,support,delivered,target):
    b=ring(support);need(b.any(),'Nonempty inner boundary');raw=np.abs(result-target_float)[0].transpose(1,2,0).astype(np.float64);quant=np.abs(delivered.astype(np.float64)-target.astype(np.float64))/255
    return {'raw_hole_L1':float(raw[support].mean()),'raw_boundary_L1':float(raw[b].mean()),'PNG_hole_L1':float(quant[support].mean()),'PNG_boundary_L1':float(quant[b].mean())}
def aggregate(rows,condition=False):
    need(bool(rows),'Nonempty aggregate')
    def avg(part):need(bool(part),'Nonempty subgroup');return {k:sum(r[k] for r in part)/len(part) for k in METRICS}
    value={'macro':avg(rows),'geometry':{kind:avg([r for r in rows if r['mask_kind']==kind]) for kind in KINDS}}
    if condition:value['condition']={name:avg([r for r in rows if r['condition']==name]) for name in sorted({r['condition'] for r in rows})}
    return value

def compare(now,old,condition=False):
    pairs=[('macro',now['macro'],old['macro'])]
    for group in ('geometry','condition') if condition else ('geometry',):
        need(set(now[group])==set(old[group]),'Compared subgroup sets exact');pairs.extend((group+'/'+key,now[group][key],old[group][key]) for key in now[group])
    failures=[{'group':label,'metric':k,'candidate':a[k],'baseline':b[k],'delta':a[k]-b[k]} for label,a,b in pairs for k in METRICS if a[k]>b[k]+1e-12]
    gain=(old['macro']['PNG_hole_L1']-now['macro']['PNG_hole_L1'])/max(old['macro']['PNG_hole_L1'],1e-12)
    return {'PNG_macro_hole_gain':gain,'PNG_macro_hole_gain_percent':100*gain,'all_raw_PNG_hole_boundary_groups_nonworse':not failures,'nonworse_failures':failures,'catastrophic_regression_stop':any(a[k]>b[k]*1.1 for _,a,b in pairs for k in METRICS),'prospective_numeric_screen_pass':gain>=.01 and not failures,'condition_groups_included':condition,'model_qualified':False}
def close_metrics(actual,reported,label):
    maxima={'raw':0.,'PNG':0.}
    for group in actual:
        pairs=[(actual[group],reported[group])] if group=='macro' else [(actual[group][k],reported[group][k]) for k in actual[group]]
        for a,b in pairs:
            for k in METRICS:
                err=abs(a[k]-b[k]);typ='PNG' if k.startswith('PNG') else 'raw';maxima[typ]=max(maxima[typ],err);need(err<=(PNG_TOL if typ=='PNG' else RAW_TOL),label+' '+k)
    return maxima

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--transport-sha',required=True);args=parser.parse_args()
    started=time.monotonic();returned=RETURN.resolve(strict=True);out=returned/'outputs';destination=returned.parent/'independent_saved_output_audit.json';need(not destination.exists(),'Preserve earlier independent receipts')
    transport_path=returned.parent/'transport_audit.json';need(sha(transport_path)==args.transport_sha,'Exact root-admitted transport receipt SHA')
    transport=read(transport_path);need(transport.get('complete') is True,'Transport receipt incomplete')
    need(sha(returned/'protocol.json')==PROTOCOL_SHA and sha(STAGE/'protocol.json')==PROTOCOL_SHA,'Exact sealed and returned protocol')
    p=read(returned/'protocol.json');need(p['format']=='completion-codeformer-same-codebook-probe-v1','Probe format');manifest=read(returned/'export_manifest.json');need(manifest['protocol_sha256']==PROTOCOL_SHA,'Manifest protocol')
    pins={r['path']:r for r in manifest['files']};need(len(pins)==len(manifest['files']),'Unique returned pins');audited_pins={};source_pins={};array_reviews=[]
    def evidence(path):
        relative=path.relative_to(returned).as_posix();need(relative in pins,'Manifest-bound evidence '+relative);row=pins[relative];need(path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Returned evidence exact '+relative);audited_pins[relative]=row;return path
    files={r['path']:r for r in p['files']};need(len(files)==254 and len(files)==len(p['files']),'Exactly254 staged payload pins')
    for relative,row in files.items():
        path=STAGE/relative;need(path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Full frozen source/asset manifest exact '+relative);source_pins[relative]=pin(path)
    need(files['scripts/cctv_completion_codeformer_same_codebook_probe_v1_vm.py']['sha256']=='7036fe80d6b942a22f7eb5bf2c662ec49a9e5e99d4b347ccb2a7b9eaa374b2f6','Exact actual worker source')
    need(files['research/completion_codeformer_same_codebook_probe_v1/probe_forward.py']['sha256']=='507101cacc6b2d42bf0ebb8e53fa1a39f2f2c2945625aaa9aad3207129912a69','Exact bridge source')
    def asset(relative,mode='RGB'):need(relative in files,'Staged source admitted');return pixels(STAGE/relative,mode)
    def arrays(path,expected,*,returned_array=True):
        if returned_array:evidence(path)
        with zipfile.ZipFile(path) as z:
            members=z.infolist();need(len(members)==len(expected) and len({v.filename for v in members})==len(members) and {v.filename for v in members}=={k+'.npy' for k in expected},'Exact numeric archive schema')
            need(all(not v.is_dir() and not v.flag_bits&1 and 0<v.file_size<=16*1024**2 and v.compress_type in (0,8) and PurePosixPath(v.filename).name==v.filename for v in members),'Safe bounded NPZ members');need(z.testzip() is None,'Full NPZ CRC')
        with np.load(path,allow_pickle=False,max_header_size=512) as z:values={k:z[k].copy() for k in z.files}
        need(all(not v.dtype.hasobject and v.dtype in (np.float32,np.int64,np.uint8) for v in values.values()),'Exact numeric only array dtypes')
        if returned_array:array_reviews.append({'path':path.relative_to(returned).as_posix(),'schema':sorted(values),'CRC_pass':True,'allow_pickle':False,'max_header_size':512,'numeric_only':True})
        return values
    oracle={};labels=[]
    for row in p['oracle_labels']:
        need(row['oracle_unavailable_clean_information'] is True and row['teacher_original_logical_sha256']=='586a8854ed7b5e3f0e202c258addb7a87de63cd3068d153741b065379ff6668f','Same original inpainting codebook labels only')
        path=STAGE/row['path'];need(sha(path)==row['sha256'] and path.stat().st_size==row['bytes'] and sha(STAGE/row['clean512'])==row['clean512_sha256'],'Exact original teacher/clean512 binding')
        z=arrays(path,{'prequant_latent','nearest_codes'},returned_array=False);codes(z['nearest_codes']);finite(z['prequant_latent'],(1,256,16,16));oracle[row['teacher_key']]=z['nearest_codes'];labels.append({'teacher_key':row['teacher_key'],'source':pin(path),'clean512_sha256':row['clean512_sha256'],'unused_prequant_latent_not_a_network_input':True})
    need(len(oracle)==16,'Exactly16 reused labels; zero regenerated teacher outputs')
    label_receipt=read(evidence(out/'oracle_label_receipt.json'));need(label_receipt['sets']==16 and label_receipt['teacher_calls']==0 and label_receipt['existing_same_codebook_labels_only'] and label_receipt['unused_prequant_latents_not_passed_to_model'] and label_receipt['source_pins']==p['oracle_labels'],'Teacher receipt exact')
    parity_summary=read(evidence(out/'ACTUAL_fresh_app_parity/summary.json'));need(parity_summary['complete'] and parity_summary['all4_saved_before_gate'] and parity_summary['tolerance']==1e-6,'Completed fresh parity scope')
    parity=[];reported_parity={r['id']:r for r in parity_summary['cases']}
    for case in p['parity_cases']:
        ident=case['id'];folder=out/'ACTUAL_fresh_app_parity';record=read(evidence(folder/(ident+'.json')));need(record==reported_parity[ident],'Parity summary row exact');z=arrays(folder/(ident+'.npz'),PARITY_SCHEMA)
        source=z['input256'];m=z['mask256'];finite(source,(1,3,256,256));finite(m,(1,1,256,256));need(np.isin(m,[0,1]).all(),'Parity binary support');support=m[0,0].astype(bool);same(png(source),asset(case['input256']),'Parity source PNG exact');same((m[0,0]*255).astype(np.uint8),asset(case['mask'],'L'),'Parity reviewed mask exact');same(z['own_input512'],z['app_input512'],'Original app input512 exact');same(z['own_codes'],z['app_codes'],'Original app selected codes exact');codes(z['own_codes']);codes(z['app_codes'])
        replay_errors=[]
        for side in ('own','app'):
            finite(z[side+'_input512'],(1,3,512,512));finite(z[side+'_raw512'],(1,3,512,512));finite(z[side+'_result256'],(1,3,256,256));same(z[side+'_result256'][:,:,~support],source[:,:,~support],'Parity exact outside float');err=float(np.max(np.abs(decode(z[side+'_raw512'],source,support)-z[side+'_result256'])));need(err<=np.finfo(np.float32).eps,'Parity raw512 replay one FP32 ulp');replay_errors.append(err)
        err=float(np.max(np.abs(z['own_result256']-z['app_result256'])));rawerr=float(np.max(np.abs(z['own_raw512']-z['app_raw512'])));need(err==record['max_raw256_error']<=1e-6 and rawerr==record['model_raw512_max_error'] and record['input512_exact'] and record['input_stride_exact'] and record['predicted_codes_exact'] and record['used_equals_predicted'],'Parity receipts match saved arrays');parity.append({'id':ident,'raw256_max_error':err,'raw512_max_error':rawerr,'input_and_codes_exact':True,'outside_float_exact':True,'raw512_replay_max_error':max(replay_errors)})
    need(len(parity)==len(reported_parity)==4,'Four fresh parity pairs')
    evaluations=[];stats={};case_rows={};predictions={};max_replay=0.;max_metric={'raw':0.,'PNG':0.};outside_bytes=0;compatibility=[];historical_report=read(evidence(out/'ACTUAL_R1_48case_compatibility.json'));historical_rows={(r['kind'],r['id']):r for r in historical_report['cases']};code_diagnostics=[]
    need(historical_report['complete'] and historical_report['pass'] and historical_report['before_any_oracle'],'Completed historical control before controls/oracles')
    for arm in p['arms']:
        for kind,key in [('monitor','monitor_cases'),('challenge','challenge_cases')]:
            folder=out/arm['filename_tag']/kind;summary=read(evidence(folder/'results.json'));cases=p[key];need(summary['arm']==arm['id'] and summary['kind']==kind and summary['w']==arm['w'] and summary['oracle_unavailable_clean_information']==arm['oracle'] and [r['id'] for r in summary['records']]==[r['id'] for r in cases],'Exact completed arm/cohort case order');rows=[]
            for case,reported in zip(cases,summary['records']):
                ident=case['id'];tag=ident+('__ORACLE_unavailable_clean_information' if arm['oracle'] else '');npz=folder/(tag+'.npz');image=folder/(tag+'.png');evidence(image);delivered=pixels(image);source8=asset(case['input256']);mask8=asset(case['mask'],'L');target=asset(case['clean256']);need(np.isin(mask8,[0,255]).all(),'Binary reviewed mask');support=mask8==255;need(int(support.sum())==case['mask_pixels'] and support.any(),'Exact active reviewed mask area')
                z=arrays(npz,SCHEMA);source=z['input256'];m=z['mask256'];result=z['result256'];raw=z['raw512'];finite(source,(1,3,256,256));finite(m,(1,1,256,256));finite(result,(1,3,256,256));finite(raw,(1,3,512,512));codes(z['predicted_codes']);codes(z['used_codes']);need(result.min()>=0 and result.max()<=1 and np.isin(m,[0,1]).all(),'Delivered range and binary support');same(png(source),source8,'Actual source byte-identical');same((m[0,0]*255).astype(np.uint8),mask8,'Actual reviewed support identical');need(float(np.max(np.abs(source[0].transpose(1,2,0)-source8.astype(np.float32)/np.float32(255))))<=np.finfo(np.float32).eps,'Canonical source floats within1ulp')
                same(result[:,:,~support],source[:,:,~support],'Exact outside float');same(delivered[~support],source8[~support],'Exact outside RGB bytes');outside_bytes+=int((~support).sum())*3;same(delivered,png(result),'Saved float-to-PNG exact');replay=decode(raw,source,support);err=float(np.max(np.abs(replay-result)));max_replay=max(max_replay,err);need(err<=np.finfo(np.float32).eps,'Saved raw512 decode one FP32 ulp');same(png(replay),delivered,'Saved raw512→PNG exact')
                need(z['arm_code'].dtype==np.uint8 and z['arm_code'].shape==() and z['arm_code'].item()==ord(arm['id']) and z['oracle_unavailable_clean_information'].dtype==np.uint8 and z['oracle_unavailable_clean_information'].shape==() and z['oracle_unavailable_clean_information'].item()==int(arm['oracle']),'Exact scalar arm/oracle flags')
                expected=oracle[case['teacher_key']]
                if arm['id']=='A':predictions[ident]=z['predicted_codes'].copy();code_diagnostics.append({'id':ident,'kind':kind,'teacher_key':case['teacher_key'],'predicted_clean_context_code_agreement':float((z['predicted_codes']==expected).mean()),'mismatched_indices':int(np.count_nonzero(z['predicted_codes']!=expected)),'indices':256})
                else:same(z['predicted_codes'],predictions[ident],'Predicted indices unchanged across all four arms')
                same(z['used_codes'],expected if arm['oracle'] else z['predicted_codes'],'Exact oracle labels or unchanged actual selected codes')
                target_float=source if np.array_equal(source8,target) else target.astype(np.float32).transpose(2,0,1)[None]/np.float32(255);values=metric(result,target_float,support,delivered,target)
                for k in METRICS:
                    delta=abs(values[k]-reported[k]);typ='PNG' if k.startswith('PNG') else 'raw';max_metric[typ]=max(max_metric[typ],delta);need(delta<=(PNG_TOL if typ=='PNG' else RAW_TOL),'Independent per-case error metric '+k)
                need(reported['id']==ident and reported['mask_kind']==case['mask_kind'] and reported['condition']==case['condition'] and reported['style']==case['style'] and reported['teacher_key']==case['teacher_key'] and reported['arm']==arm['id'] and reported['w']==arm['w'] and reported['oracle_unavailable_clean_information']==arm['oracle'] and reported['not_deployable']==(arm['id']!='A') and reported['outside_exact'] and reported['PNG_sha256']==sha(image) and reported['PNG']==image.relative_to(out).as_posix(),'Exact actual row metadata/output binding')
                need(reported['predicted_used_code_agreement']==float((z['predicted_codes']==z['used_codes']).mean()),'Saved code agreement receipt exact')
                row={'id':ident,'kind':kind,'arm':arm['id'],'w':arm['w'],'oracle_unavailable_clean_information':arm['oracle'],'mask_kind':case['mask_kind'],'condition':case['condition'],'style':case['style'],'teacher_key':case['teacher_key'],'NPZ':pin(npz),'PNG':pin(image),'outside_FP32_PNG_exact':True,'raw512_replay_max_error':err,'raw512_replay_PNG_different_channels':0,'predicted_codes_match_A':True,'used_codes_match_declared_source':True,'predicted_used_code_agreement':reported['predicted_used_code_agreement'],**values}
                if case['style']=='GRAY':row['GRAY_raw_mean_channel_spread']=float(np.ptp(result[0].transpose(1,2,0),axis=-1)[support].mean());row['GRAY_PNG_mean_channel_spread']=float(np.ptp(delivered.astype(np.int16),axis=-1)[support].mean())
                rows.append(row);evaluations.append(row);case_rows[(arm['id'],kind,ident)]=row
                if arm['id']=='A':
                    old=next(r for r in p['historical_control']['initial_rows'][kind] if r['id']==ident);zold=arrays(STAGE/old['result256'],{'result256'},returned_array=False);finite(zold['result256'],(1,3,256,256));rawerr=float(np.max(np.abs(zold['result256']-result)));exact=sha(image)==sha(STAGE/old['PNG']);vm=historical_rows[(kind,ident)];need(rawerr<=1e-6 and exact and vm['raw256_max_error']==rawerr and vm['PNG_bytes_exact']==exact,'All48 actualA historical R1 raw/PNG match');compatibility.append({'id':ident,'kind':kind,'raw256_max_error':rawerr,'PNG_bytes_exact':exact})
            value=aggregate(rows,condition=kind=='challenge');close_metrics(value,summary['metrics'],'Completed aggregate metrics');stats[(arm['id'],kind)]=value
        arm_receipt=read(evidence(out/('arm_'+arm['id']+'_complete.json')));need(arm_receipt['complete'] and arm_receipt['arm']==arm and arm_receipt['oracle_unavailable_clean_information']==arm['oracle'],'Arm completion contract')
        print(json.dumps({'audited_arm':arm['id'],'audited_outputs':len(evaluations),'outside_RGB_bytes_exact':outside_bytes}),flush=True)
    need(len(compatibility)==48 and len(evaluations)==192 and len(array_reviews)==196,'Complete actual saved outputs and historical comparison')
    comparisons={};percase={};pairs=[('B','A'),('C','A'),('D','B'),('D','A'),('D','C'),('C','B')]
    for candidate,baseline in pairs:
        name=candidate+'_vs_'+baseline;comparisons[name]={};percase[name]={}
        for kind,key in [('monitor','monitor_cases'),('challenge','challenge_cases')]:
            comparisons[name][kind]=compare(stats[(candidate,kind)],stats[(baseline,kind)],condition=kind=='challenge');changes=[]
            for case in p[key]:
                a=case_rows[(candidate,kind,case['id'])];b=case_rows[(baseline,kind,case['id'])];changes.append({'id':case['id'],'mask_kind':case['mask_kind'],'condition':case['condition'],'PNG_hole_gain_percent':100*(b['PNG_hole_L1']-a['PNG_hole_L1'])/max(b['PNG_hole_L1'],1e-12),'metric_deltas':{k:a[k]-b[k] for k in METRICS},'PNG_hole_worse':a['PNG_hole_L1']>b['PNG_hole_L1']+1e-12,'PNG_boundary_worse':a['PNG_boundary_L1']>b['PNG_boundary_L1']+1e-12})
            percase[name][kind]={'cases':changes,'PNG_hole_worse_count':sum(r['PNG_hole_worse'] for r in changes),'PNG_boundary_worse_count':sum(r['PNG_boundary_worse'] for r in changes)}
    final=read(evidence(out/'results.json'));need(final['complete'] and final['protocol_sha256']==PROTOCOL_SHA and final['outputs_saved']==192 and final['oracle_outputs_explicitly_labeled']==96 and final['all497_state_before_after_identical'] and final['diagnostic_only'] and not final['new_model_created'] and not final['checkpoint_saved'] and not final['model_qualified'] and not final['automatic_app_promotion'] and final['oracle_not_deployable_or_identity_recovery'] and final['same_codebook_target_not_mathematical_upper_bound'],'Actual diagnostic-only final contract')
    expected_counts={'forward_slots':200,'actual_whole_model_forward_calls':200,'backward_calls':0,'gradient_queries':0,'optimizer_objects':0,'optimizer_steps':0,'teacher_neural_calls':0};need(final['counts']==expected_counts and final['typed_forward_slots']=={'parity':8,'A':48,'B':48,'C':48,'D':48} and final['optimizer_steps']==final['backward_calls']==0,'Exact200 calls zero training')
    gate_ambiguities=[]
    for arm in ('B','C','D'):
        for kind in ('monitor','challenge'):
            actual=comparisons[arm+'_vs_A'][kind];vm=final['diagnostic_comparisons_vs_A'][arm][kind];need(abs(actual['PNG_macro_hole_gain']-vm['PNG_macro_hole_gain'])<=PNG_TOL,'Final PNG gain exactly independently recomputed')
            for field in ('prospective_numeric_screen_pass','all_raw_PNG_hole_boundary_groups_nonworse','catastrophic_regression_stop'):
                if actual[field]!=vm[field]:gate_ambiguities.append({'arm':arm,'kind':kind,'field':field,'VM':vm[field],'independent':actual[field],'RAW_float_reduction_difference_may_affect_1e-12_edge':True})
    need(not gate_ambiguities,'Reported versus independently recomputed subgroup decisions differ; inspect exact endpoints')
    need(final['ordinary_app_parity']==parity_summary['cases'] and final['actual_R1_historical_compatibility']==historical_report['cases'],'Final parity/history exactly retained')
    returned_npz={path.relative_to(returned).as_posix() for path in out.rglob('*.npz')};need(returned_npz=={r['path'] for r in array_reviews},'Every retained NPZ independently audited');returned_png={path.relative_to(returned).as_posix() for path in out.rglob('*.png')};need(returned_png=={Path(r['PNG']['path']).relative_to(returned).as_posix() for r in evaluations},'Every192 returned PNG audited')
    need(not any('teacher_labels' in path.parts for path in out.rglob('*')) and not any(s.lower() in ('.pth','.pt') for path in out.rglob('*') for s in path.suffixes),'No teacher/checkpoint output silently generated')
    for relative,row in audited_pins.items():need((returned/relative).stat().st_size==row['bytes'] and sha(returned/relative)==row['sha256'],'Returned evidence unchanged during pixel audit')
    for relative,row in source_pins.items():need(sha(STAGE/relative)==row['sha256'],'Frozen source unchanged during pixel audit')
    need(not any(k=='torch' or k.startswith('torch.') or k.startswith('third_party.codeformer') for k in sys.modules),'No Torch/model runtime imported')
    report={'format':'completion-codeformer-same-codebook-probe-v1-independent-saved-pixel-index-audit-v1','complete':True,'independent':True,'audited_UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(Path(__file__)),'transport_admission':pin(transport_path),'protocol':pin(returned/'protocol.json'),'evaluated_outputs':192,'all_retained_NPZ':196,'all_retained_PNG':192,'oracle_outputs_explicitly_labeled':96,'all_NPZ_schema_CRC_numeric_finite_pass':True,'all_raw512_to_PNG_exact':True,'all_saved_float256_to_PNG_exact':True,'raw512_to_result_max_FP32_error':max_replay,'outside_FP32_PNG_all_exact':True,'outside_source_RGB_bytes_exact':outside_bytes,'parity_pairs':parity,'historical_R1_actualA_48case_compatibility':compatibility,'reused_original_same_codebook_label_sets':labels,'actual_predicted_codes_unchanged_across_arms':True,'selected_actual_codes_equal_predicted':True,'selected_oracle_codes_equal_frozen_original_clean_context_labels':True,'code_agreement_diagnostics':code_diagnostics,'independent_cohort_metrics':{arm:{kind:stats[(arm,kind)] for kind in ('monitor','challenge')} for arm in 'ABCD'},'independent_comparisons':comparisons,'independent_percase_comparisons':percase,'VM_metric_maximum_reduction_differences':max_metric,'VM_screen_decisions_independently_verified':True,'quality_thresholds_unchanged':{'prospective_PNG_macro_hole_reduction_minimum':.01,'all_raw_PNG_hole_boundary_subgroup_nonworse_tolerance':1e-12,'RAW_metric_recompute_absolute_tolerance':RAW_TOL,'PNG_metric_recompute_absolute_tolerance':PNG_TOL,'RAW512_decode_absolute_FP32_ulp':float(np.finfo(np.float32).eps)},'all_outputs':evaluations,'array_reviews':array_reviews,'staged_source_and_asset_pins':source_pins,'frozen_model_state_audit_owned_separately':True,'scope':'48 paired adaptive TRAIN cases (32 clear monitor plus16 fixed challenges), reviewed assisted masks only. No public/native CCTV, FINAL review, automatic-mask or identity evidence. C/D oracle codes require unavailable clean information and cannot be deployed. B is an untrained inference control and requires visual/user/further acceptance before adoption. Hole-L1 reductions are NOT restoration feature structure gains.','no_teacher_network_or_model_replay':True,'visual_quality_review_performed_by_auditor':False,'new_model_created':False,'model_qualified':False,'automatic_app_promotion':False,'execution':{'Torch_imported':False,'neural_calls':0,'gradient_queries':0,'optimizer_steps':0,'VM_calls':0},'checks':CHECKS,'seconds':time.monotonic()-started}
    with destination.open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'complete':True,'report':pin(destination),'checks':CHECKS,'outputs':192,'NPZ':196,'outside_RGB_bytes_exact':outside_bytes,'comparisons':comparisons,'model_qualified':False,'seconds':report['seconds']}),flush=True)
if __name__=='__main__':main()

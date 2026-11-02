"""Independent QA of root's saved-code geometric diagnosis and PNG closure.
Reads only exact admitted saved numeric arrays/PNGs. No Torch/model/VM calls.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math,os,zipfile,io
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
RET=ROOT/'outputs/completion_codeformer_predictor_ce_v1_return'
RUN=RET/'cctv_completion_codeformer_predictor_ce_v1_results'
STAGE=ROOT/'outputs/completion_codeformer_predictor_ce_v1_packet/draft_001/cctv_completion_codeformer_predictor_ce_v1_vm'
CHECKS=0

def need(ok,message):
 global CHECKS
 CHECKS+=1
 if not bool(ok):raise ValueError(message)

def io_path(path):
 p=Path(path).resolve();return Path('\\\\?\\'+str(p)) if os.name=='nt' and not str(p).startswith('\\\\?\\') else p

def sha(path):
 h=hashlib.sha256()
 with io_path(path).open('rb') as stream:
  for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def read(path):return json.loads(io_path(path).read_text(encoding='utf-8-sig'))
def pin(path):return {'path':str(Path(path).resolve()),'bytes':io_path(path).stat().st_size,'sha256':sha(path)}

def rgb(path,mode='RGB'):
 with Image.open(io_path(path)) as im:need(im.format=='PNG' and im.mode==mode and im.size==(256,256),'Exact PNG schema');return np.array(im)

def code(path,name):
 with zipfile.ZipFile(io_path(path)) as z:
  info=z.getinfo(name+'.npy');need(info.file_size<4096 and not info.flag_bits&1 and info.compress_type in (0,8),'Bounded numeric code member');v=np.load(io.BytesIO(z.read(info)),allow_pickle=False,max_header_size=512)
 need(v.dtype==np.int64 and v.shape==(1,256) and ((v>=0)&(v<512)).all(),'Exact512code shape/range');return v[0]

def close(a,b,msg):need(a==b if a is None or b is None else math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-14),msg)

def summarize(rows):
 result={}
 for region in ('all','touched','majority','fully_hidden','visible','crossing'):
  have=[r['regions'][region] for r in rows if r['regions'][region]['tokens']]
  tokens=sum(x['tokens'] for x in have)
  result[region]={'cases':len(have),'tokens':tokens,'mean_case_agreement':sum(x['agreement'] for x in have)/len(have) if have else None,'pooled_token_agreement':sum(x['correct'] for x in have)/tokens if have else None,'mean_case_changed_fraction_from_R1':sum(x['changed_from_R1_fraction'] for x in have)/len(have) if have else None,'pooled_changed_tokens':sum(x['changed_from_R1_tokens'] for x in have)}
 return result

def main():
 geometrypath=RET/'root_saved_code_geometry_diagnosis.json';closurepath=RET/'root_endpoint_and_preservation_closure.json';pixelpath=RET/'independent_saved_output_audit.json'
 need(sha(geometrypath)=='647bfa63a9f7aaa62d18cc7e494a5bba5ccf5468d224e3293d5370a469a7fca8','Exact root geometry artifact');need(sha(closurepath)=='a8f680bab3455ff2e459f01290caf8c46c948af97df5b855b3951eab561b29a3','Exact root PNG closure');need(sha(RET/'transport_audit.json')=='60bcc196d165e082010d67d1a62708792cf1f08dfa732921a58a1d65f13408c0','Admitted transport');need(sha(RUN/'protocol.json')==sha(STAGE/'protocol.json')=='15aa3e1478a67c120b2fcb6af50ecae58d0558eb4e65cef473d0339572bf084b','Exact protocol')
 p=read(RUN/'protocol.json');d=read(geometrypath);closure=read(closurepath);pixel=read(pixelpath);files={x['path']:x for x in p['files']};labels={x['teacher_key']:x for x in p['cached_labels']};rootrows={(r['cohort'],r['id'],r['cycle']):r for r in d['records']};need(len(rootrows)==len(d['records'])==192 and d['snapshots']==[0,8,16,32] and d['paired_cases']==48 and d['saved_code_observations']==192,'All48cases4snapshots scope');independent=[]
 for cohort,key in [('monitor','monitor_cases'),('challenge','challenge_cases')]:
  for case in p[key]:
   maskpath=STAGE/case['mask'];need(sha(maskpath)==files[case['mask']]['sha256'],'Exact reviewed mask pin');mask=rgb(maskpath,'L');need(np.isin(mask,[0,255]).all(),'Binary reviewed mask');support=mask==255
   # Deliberately use explicit row/column slices rather than root's reshape/mean implementation.
   counts=np.array([np.count_nonzero(support[y:y+16,x:x+16]) for y in range(0,256,16) for x in range(0,256,16)])
   selections={'all':np.full(256,True),'touched':counts>0,'majority':counts>=128,'fully_hidden':counts==256,'visible':counts==0,'crossing':(counts>0)&(counts<256)}
   need(np.array_equal(selections['touched']|selections['visible'],selections['all']) and not (selections['touched']&selections['visible']).any() and np.array_equal(selections['fully_hidden']|selections['crossing'],selections['touched']),'Disjoint geometric token classification')
   labrow=labels[case['teacher_key']];labpath=STAGE/labrow['path'];need(sha(labpath)==labrow['sha256'],'Exact original samebook labels');target=code(labpath,'nearest_codes');basepath=RUN/'outputs/u0000'/cohort/(case['id']+'.npz');basecode=code(basepath,'predicted_codes');basepng=rgb(RUN/'outputs/u0000'/cohort/(case['id']+'.png'));previous=basecode
   for cycle in (0,8,16,32):
    row=rootrows[(cohort,case['id'],cycle)];path=RUN/('outputs/u%04d'%cycle)/cohort/(case['id']+'.npz');pred=code(path,'predicted_codes');png=rgb(RUN/('outputs/u%04d'%cycle)/cohort/(case['id']+'.png'));need(row['predicted_codes_file']==pin(path) and row['fixed_label_file']==pin(labpath),'Root actual code/label file pins');need(row['mask_kind']==case['mask_kind'] and row['style']==case['style'] and row['condition']==case['condition'],'Root fixed row metadata');regions={}
    for name,chosen in selections.items():
     number=int(np.count_nonzero(chosen));correct=int(np.count_nonzero(pred[chosen]==target[chosen]));changed=int(np.count_nonzero(pred[chosen]!=basecode[chosen]));regions[name]={'tokens':number,'correct':correct,'agreement':correct/number if number else None,'changed_from_R1_tokens':changed,'changed_from_R1_fraction':changed/number if number else None};need(row['regions'][name]==regions[name],'Independent exact token geometry/codes '+case['id']+' '+name)
    need(row['changed_tokens_from_previous_snapshot']==np.count_nonzero(pred!=previous),'Previous snapshot code-change count');previous=pred
    change=float(np.abs(png[support].astype(np.int16)-basepng[support].astype(np.int16)).mean()/255);close(change,row['PNG_mean_absolute_change_within_mask_from_R1'],'PNG masked delta independent replay');need(np.array_equal(png[~support],basepng[~support]),'Mask outside PNG exact')
    independent.append({'cohort':cohort,'id':case['id'],'cycle':cycle,'mask_kind':case['mask_kind'],'regions':regions})
 for cohort in ('monitor','challenge'):
  for cycle in (0,8,16,32):
   subset=[r for r in independent if r['cohort']==cohort and r['cycle']==cycle];agg=d['aggregates'][cohort][str(cycle)];expected=summarize(subset)
   for region,metrics in expected.items():
    for k,v in metrics.items():close(v,agg['macro'][region][k],'Macro token metrics exact')
   for kind in {r['mask_kind'] for r in subset}:
    expected=summarize([r for r in subset if r['mask_kind']==kind])
    for region,metrics in expected.items():
     for k,v in metrics.items():close(v,agg['geometry'][kind][region][k],'Pergeometry token metrics exact')
 transitions=[]
 for cohort in ('monitor','challenge'):
  byid={r['id']:r for r in d['records'] if r['cohort']==cohort and r['cycle']==0};final=[r for r in d['records'] if r['cohort']==cohort and r['cycle']==32];item={'cohort':cohort,'cases':len(final),'code_touched_agreement_improved_and_PNG_worse':[],'code_touched_agreement_worse_and_PNG_better':[]}
  for r in final:
   delta=r['regions']['touched']['agreement']-byid[r['id']]['regions']['touched']['agreement'];gain=r['PNG_hole_error_reduction_percent_from_R1']
   if delta>0 and gain<0:item['code_touched_agreement_improved_and_PNG_worse'].append(r['id'])
   if delta<0 and gain>0:item['code_touched_agreement_worse_and_PNG_better'].append(r['id'])
  transitions.append(item)
 need(transitions==d['individual_cross_endpoint_disagreements'],'Exact cross-endpoint disagreement lists');need(d['local_neural_calls']==d['optimizer_updates']==d['VM_calls']==0 and not d['model_qualified'] and d['R1_remains_selected'],'Diagnostic status stays separate from qualified model');need(len(d['limits'])==5 and any('not the encoder receptive fields' in x for x in d['limits']) and any('not face identity' in x for x in d['limits']) and any('not candidate selection' in x for x in d['limits']) and any('No causal attribution' in x for x in d['limits']),'Explicit correct inference/selection limits')
 # Root closure's PNG values are independently cross-checked against all saved target/source pixels.
 pg={(r['kind'],r['id'],r['cycle']):r for r in closure['paired_PNG_records']};need(len(pg)==192 and closure['PNG_outputs_replayed']==256 and closure['paired_metric_observations']==192 and closure['exact_outside_RGB_bytes']==pixel['outside_source_RGB_bytes_exact']==44001540,'Complete PNG/pixel scope')
 for cohort,key in [('monitor','monitor_cases'),('challenge','challenge_cases')]:
  for case in p[key]:
   m=rgb(STAGE/case['mask'],'L')==255;target=rgb(STAGE/case['clean256']);source=rgb(STAGE/case['input256']);edge=m.copy()
   # Independent boundary via OR-shifts of visible pixels; avoid sliding-window implementation.
   inv=(~m);expanded=np.zeros_like(m)
   for dy in range(-3,4):
    for dx in range(-3,4):
     ys=slice(max(0,dy),min(256,256+dy));xs=slice(max(0,dx),min(256,256+dx));expanded[ys,xs]|=inv[slice(max(0,-dy),min(256,256-dy)),slice(max(0,-dx),min(256,256-dx))]
   boundary=m&expanded;need(boundary.any(),'Nonempty reviewed support edge')
   for cycle in (0,8,16,32):
    png=rgb(RUN/('outputs/u%04d'%cycle)/cohort/(case['id']+'.png'));need(np.array_equal(png[~m],source[~m]),'All192pairedPNGoutsideexact');delta=np.abs(png.astype(np.int16)-target.astype(np.int16));row=pg[(cohort,case['id'],cycle)];close(float(delta[m].mean()/255),row['PNG_hole_L1'],'Root hole PNG direct remeasurement');close(float(delta[boundary].mean()/255),row['PNG_boundary_L1'],'Root boundary PNG independent OR-shifts');gr=rootrows[(cohort,case['id'],cycle)];close(row['PNG_hole_L1'],gr['PNG_hole_L1'],'Root geometric PNG metric joins root measured PNG')
    base=pg[(cohort,case['id'],0)];close(100*(base['PNG_hole_L1']-row['PNG_hole_L1'])/base['PNG_hole_L1'],gr['PNG_hole_error_reduction_percent_from_R1'],'Root geometric PNG percent direct replay')
    if case['style']=='GRAY':close(float(np.ptp(png.astype(np.int16),axis=-1)[m].mean()),row['GRAY_PNG_mean_channel_spread'],'Root gray pixel spread')
 for cycle in (8,16,32):
  for cohort in ('monitor','challenge'):
   c=closure['separate_cohort_comparisons'][str(cycle)][cohort];ind=pixel['independent_comparisons_vs_R1'][str(cycle)][cohort];close(c['PNG_hole_error_reduction_percent'],ind['PNG_macro_hole_gain_percent'],'Root vs independent pixel macro endpoint')
 need(closure['candidate_adopted'] is False and closure['neural_calls']==closure['VM_calls']==0 and closure['model_qualified'] is False,'Root closure nonadoption status');need(len(closure['R1_app_model_assets_exact'])==22,'Root22 app/model pins')
 transport=read(RET/'transport_audit.json');need(closure['payload_files_rehashed']==len(transport['files'])==835 and any(r['path']=='export_manifest.json' for r in transport['files']),'Explain root count-label ambiguity honestly')
 notes=[{'artifact':'root_endpoint_and_preservation_closure.json','field':'payload_files_rehashed','value':835,'precise_meaning':'835 regular files including export_manifest.json; 834 manifest payload files. Count arithmetic is correct, field name is imprecise. Preserve original root artifact; do not report 835 manifest payloads.'}]
 report={'format':'independent-predictor-CE-v1-root-geometric-and-PNG-diagnosis-QA','complete':True,'independent':True,'UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(__file__),'root_geometry':pin(geometrypath),'root_PNG_closure':pin(closurepath),'independent_pixel_audit':pin(pixelpath),'checks_passed':CHECKS,'all48cases_4snapshots_192code_observations_recomputed':True,'geometric_cells_independently_counted_with_explicit_pixel_slices':True,'all6regions_exact_per_case':True,'allmacro_and_fourgeometry_aggregates_exact':True,'all192pairedPNG_hole_boundary_and_gray_metrics_independently_replayed':True,'all3snapshot_cohort_gains_equal_independent_pixel_audit':True,'cross_endpoint_disagreement_lists_exact':True,'limitations_correct':True,'factual_count_clarifications':notes,'no_scientific_or_endpoint_errors_found':True,'no_receptivefield_identity_causal_or_candidate_selection_claim':True,'local_actions':{'Torch_imported':False,'neural_calls':0,'gradients':0,'optimizer_steps':0,'VM_calls':0,'returned_code_executed':False},'model_qualified':False}
 dest=Path(__file__).parent/'independent_root_saved_diagnosis_QA.json'
 with dest.open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'complete':True,'receipt':pin(dest),'checks':CHECKS,'clarifications':notes},indent=2),flush=True)
if __name__=='__main__':main()

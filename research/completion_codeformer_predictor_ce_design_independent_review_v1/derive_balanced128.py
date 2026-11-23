"""Deterministic balanced128-case subset from exact retained256FIT pool; noNN."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parents[2];SOURCE=ROOT/'outputs/completion_codeformer_delivered_objective_v4_packet/draft_001/cctv_completion_codeformer_delivered_objective_v4_vm/protocol.json';OUT=ROOT/'research/completion_codeformer_predictor_ce_design_independent_review_v1/balanced128_subset.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok,msg):
 if not ok:raise ValueError(msg)
def pin(p):return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)}
def main():
 need(sha(SOURCE)=='8bbb3dd611e95cfbce76e7e6426ffdadb31ce23dadb1dfdcd0246e501213069d','Exact frozen pool protocol');p=json.loads(SOURCE.read_text());pool=p['fit_cases'];files={r['path']:r for r in p['files']};refs=sorted({r['reference_id'] for r in pool});need(len(refs)==32 and len(pool)==256,'Expected full pool');allgeometry=('lower_face','eye_band','image_left_lens','cheek_object');selected=[];roles={}
 grayconditions=('strict_gray_clear','strict_gray_mild_degraded');rgbconditions=('RGB_mild_degraded','RGB_clear_irregular')
 for i,ref in enumerate(refs):
  cases=[r for r in pool if r['reference_id']==ref];need(len(cases)==8,'Eight fixed cases/ref');gray=next(r for r in cases if r['condition']==grayconditions[i%2]);rgb=next(r for r in cases if r['condition']==rgbconditions[(i//2)%2]);need(gray['mask_kind']!=rgb['mask_kind'],'Chosen augmentation geometries unique');anchors=sorted([r for r in cases if r['role']=='fit_anchor' and r['mask_kind'] not in (gray['mask_kind'],rgb['mask_kind'])],key=lambda r:allgeometry.index(r['mask_kind']));need(len(anchors)==2,'Two complementary clear anchors')
  role=[anchors[0],anchors[1],gray,rgb];need({r['mask_kind'] for r in role}==set(allgeometry),'Oneeachgeometry/ref');roles[ref]=role;selected.extend(role)
 schedule=[]
 for group in range(8):
  grouprefs=refs[group*4:(group+1)*4]
  for turn in range(4):schedule.append([roles[ref][(turn+j)%4]['id'] for j,ref in enumerate(grouprefs)])
 byid={r['id']:r for r in selected};flat=[i for row in schedule for i in row];need(len(selected)==len(set(flat))==len(flat)==128 and set(flat)==set(byid),'128uniqueexactonce');need(len(schedule)==32,'32four-casecycles')
 for row in schedule:
  cases=[byid[i] for i in row];need(len({r['reference_id'] for r in cases})==4 and sum(r['role']=='fit_anchor' for r in cases)==2 and sum(r['style']=='GRAY' for r in cases)==1 and sum(r['role']=='fit_augmented' and r['style']=='RGB' for r in cases)==1,'Everycycle4refs2anchors1gray1rgbaug')
 counts={k:dict(Counter(r[k] for r in selected)) for k in ('role','style','mask_kind','condition','reference_id')};need(counts['role']=={'fit_anchor':64,'fit_augmented':64} and counts['style']=={'RGB':96,'GRAY':32} and set(counts['mask_kind'].values())=={32} and set(counts['reference_id'].values())=={4},'Balanced totals');need(counts['condition']['original_clear_anchor']==64 and all(counts['condition'][c]==16 for c in (*grayconditions,*rgbconditions)),'Augconditions16each');need(len({r['teacher_key'] for r in selected})==64 and not({r['reference_id'] for r in selected}&{r['reference_id'] for r in p['monitor_cases']+p['challenge_cases']}),'All64FITlabels consumed; monitor disjoint')
 data={}
 for r in selected:
  for field in ('input256','clean256','clean512','mask'):
   row=files[r[field]];q=SOURCE.parent/r[field];need(q.stat().st_size==row['bytes'] and sha(q)==row['sha256'],'Existing asset drift');data[r[field]]=row
 report={'format':'independent-predictor-CE-balanced128-retained-subset-design-v1','complete':True,'independent':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'deriver':pin(Path(__file__)),'source_protocol':pin(SOURCE),'source_full_pool_cases':256,'retained_full_pool_modified':False,'subset_cases':128,'references':32,'teacher_keys_consumed':64,'deterministic_selection':'Sort32FITreferenceIDs. Atreferenceindexi choose graycondition[i%2] and RGBcondition[(i//2)%2]; eachconditionhasexactlyoneoriginalaugmentation/ref. Choose clearanchor cases in complementary2maskgeometries. Keep selectedexistingassets unchanged.','deterministic_schedule':'Eight groups of four sortedreferenceIDs. For turn0..3, referencepositionj takes role[(turn+j)%4] where roles are [anchor in earliergeometryorder,anchor in latergeometryorder,grayaug,RGBaug]. Each32cycle has4distinctrefs/2anchors/1grayaug/1RGBaug.','cases_in_reference_role_order':selected,'schedule':schedule,'schedule_flat_ids':flat,'balanced_counts':counts,'all_cases_drawn_from_existing_FIT_pool_exact':True,'every_reference_oneeach4geometries':True,'every_selected_case_once':True,'monitor_challenge_FINAL_never_optimized':True,'pinned_existing_assets':list(data.values()),'optimizer_steps_prospective':32,'backwards_prospective':128,'all256FITcases_used_in_this_pilot':False,'full256pool_preserved_for_other_future_designs':True,'subset_selection_uses_no_monitor_or_quality_score':True,'local_actions':{'Torch_imported':False,'neural_calls':0,'gradients':0,'optimizer_objects_or_steps':0,'VM_calls':0,'new_training_targets':0},'runtime_or_packet_admitted':False,'model_qualified':False}
 need(not OUT.exists(),'Preserve prior subset');OUT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({'complete':True,'subset':pin(OUT),'counts':{k:v for k,v in counts.items() if k!='reference_id'},'steps':32,'cases':128,'labels':64},indent=2))
if __name__=='__main__':main()

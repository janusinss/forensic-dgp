"""Independent saved-only V3 objective/design review; no Torch or training code."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math,sys,time
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
RET=ROOT/'outputs/completion_codeformer_fusion_only_v3_return'
RUN=RET/'cctv_completion_codeformer_fusion_only_v3_results'
STAGE=ROOT/'outputs/completion_codeformer_fusion_only_v3_packet/draft_002/cctv_completion_codeformer_fusion_only_v3_vm'
DIAG=ROOT/'outputs/completion_codeformer_v3_objective_alignment_v1_r1/analysis.json'
OUT=ROOT/'outputs/completion_codeformer_v3_next_objective_independent_review_v1'
CHECKS=0

def need(ok,msg):
 global CHECKS
 CHECKS+=1
 if not ok:raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)}
def inner7(mask):
 # Inner ring; off-image support remains true, matching max-pool padding.
 padded=np.pad(mask,3,constant_values=True);eroded=np.ones_like(mask,dtype=bool)
 for y in range(7):
  for x in range(7):eroded &= padded[y:y+mask.shape[0],x:x+mask.shape[1]]
 return mask & ~eroded

def main():
 started=time.monotonic();need(not OUT.exists(),'Preserve existing review')
 expected={'diagnosis':(DIAG,'89396f99bb21cd57743e3549e201ae8a12f0a514cff0baf63d74421923144a51'),'visual':(RET/'root_visual_review.json','44d478963a56a16f09f6f6050257957132d6bba98494e3c352f5d72ccba90f24'),'saved_output':(RET/'independent_saved_output_audit.json','9788881c412173e9f564aaf512859b07179a2977089d9ffd878cb044a3da49a0'),'state':(RET/'independent_state_audit.json','52d2c3040b050bd544231ba0ea1ed577444770f6b88655fc3ea081355909a365')}
 sources={}
 for key,(p,h) in expected.items():need(sha(p)==h,'Controlling receipt changed '+key);sources[key]=pin(p)
 diagnosis=read(DIAG);p=read(RUN/'protocol.json');saved=read(expected['saved_output'][0]);visual=read(expected['visual'][0]);state=read(expected['state'][0]);manifest={r['path']:r for r in read(RUN/'export_manifest.json')['files']};sourcefiles={r['path']:r for r in p['files']}
 need(p['contract']['selection_min_macro_PNG_hole_gain']==.01 and p['contract']['nonworse_tolerance']==1e-12,'Existing gates differ')
 need(visual['decision']=='REJECT_V3_REPLACEMENT_RETAIN_R1' and visual['sheets_reviewed']==20 and saved['all_numeric_screen_endpoints_pass_if_available'] is False,'Quality evidence status differs')
 need(state['state_status']['valid_new_recovery_checkpoint'] is True and state['model_state']['frozen_bitexact_to_R1']==443 and state['optimizers']['code']['step_after']==32 and state['optimizers']['fusion']['step_after']==96,'Independent state status differs')
 records={(r['kind'],r['cycle'],r['id']):r for r in diagnosis['all_case_measurements']};delivered={(r['kind'],r['cycle'],r['id']):r for r in saved['all_outputs']};checked=[];target_cache={};input_pins={};maxdev=0.
 for kind,key in [('monitor','monitor_cases'),('challenge','challenge_cases')]:
  for cycle in (0,64):
   for case in p[key]:
    path=RUN/('outputs/u%04d'%cycle)/kind/(case['id']+'.npz');rel=path.relative_to(RUN).as_posix();need(sha(path)==manifest[rel]['sha256'],'Returned array hash differs');input_pins[rel]=pin(path)
    target=STAGE/case['clean512']
    if case['clean512'] not in target_cache:
     need(sha(target)==sourcefiles[case['clean512']]['sha256'],'HQ target hash differs')
     with Image.open(target) as im:need(im.size==(512,512) and im.mode=='RGB','Target schema');target_cache[case['clean512']]=np.array(im,dtype=np.float32).transpose(2,0,1)[None]/np.float32(255)
    with np.load(path,allow_pickle=False) as z:raw=z['raw512'].copy();support=z['mask256'][0,0].astype(bool)
    need(raw.shape==(1,3,512,512) and raw.dtype==np.float32 and np.isfinite(raw).all(),'Saved raw invalid')
    hole=np.repeat(np.repeat(support,2,axis=0),2,axis=1);boundary=inner7(hole);pred=(raw+np.float32(1))/np.float32(2);error=np.abs(pred-target_cache[case['clean512']])[0].transpose(1,2,0).astype(np.float64);clipped=np.abs(np.clip(pred,0,1)-target_cache[case['clean512']])[0].transpose(1,2,0).astype(np.float64)
    measures={'HQ512_hole_L1_unclamped':float(error[hole].mean()),'HQ512_boundary_L1_unclamped':float(error[boundary].mean()),'HQ512_hole_L1_clipped':float(clipped[hole].mean())};measures['HQ512_training_sum']=measures['HQ512_hole_L1_unclamped']+measures['HQ512_boundary_L1_unclamped']
    recorded=records[(kind,cycle,case['id'])]
    for name,v in measures.items():dev=abs(v-recorded[name]);need(dev<=1e-12,'Independent HQ measure differs '+name);maxdev=max(dev,maxdev)
    for name in ('raw_hole_L1','raw_boundary_L1','PNG_hole_L1','PNG_boundary_L1'):need(recorded[name]==delivered[(kind,cycle,case['id'])][name],'Diagnosis delivered endpoint differs')
    checked.append({'kind':kind,'cycle':cycle,'id':case['id'],'mask_kind':case['mask_kind'],'condition':case['condition'],**measures,'PNG_hole_L1':recorded['PNG_hole_L1']})
 need(len(checked)==96,'Expected48paired casesx2 endpoints')
 recomputed={}
 def gain(now,old):return 100*(old-now)/old
 for kind in ('monitor','challenge'):
  rows=[r for r in checked if r['kind']==kind];groups={'macro':lambda r:True}
  groups.update({'geometry:'+k:(lambda r,k=k:r['mask_kind']==k) for k in sorted({r['mask_kind'] for r in rows})})
  if kind=='challenge':groups.update({'condition:'+k:(lambda r,k=k:r['condition']==k) for k in sorted({r['condition'] for r in rows})})
  recomputed[kind]={}
  for group,choose in groups.items():
   cohorts={i:[r for r in rows if r['cycle']==i and choose(r)] for i in (0,64)};gains={}
   for metric in ('HQ512_hole_L1_unclamped','HQ512_boundary_L1_unclamped','HQ512_training_sum','HQ512_hole_L1_clipped','PNG_hole_L1'):
    old=float(np.mean([r[metric] for r in cohorts[0]],dtype=np.float64));new=float(np.mean([r[metric] for r in cohorts[64]],dtype=np.float64));g=gain(new,old);gains[metric]=g
    source=diagnosis['terminal_changes'][kind]['macro'] if group=='macro' else diagnosis['terminal_changes'][kind][group.split(':',1)[0]][group.split(':',1)[1]]
    need(math.isclose(g,source[metric]['gain_percent'],rel_tol=1e-10,abs_tol=1e-10),'Aggregate arithmetic differs')
   recomputed[kind][group]=gains
 need(all(g['HQ512_training_sum']>0 for k in recomputed.values() for g in k.values()),'Claim all aggregate totals improve differs')
 # State/receipt clipping check is independent of any hypothetical new gradients.
 clipping=state['training']['gradient_clipping'];need(clipping['clipped_steps']==0 and clipping['preclip_max']<1,'Clipping did not constrain this finite route')
 need(all(x['changes']['HQ_hole_improves_while_delivered_PNG_hole_worsens'] is False for x in diagnosis['terminal_case_changes']),'Do not conflate aggregate flip with individual case flip')
 need(not any(n=='torch' or n.startswith('torch.') for n in sys.modules),'Torch unexpectedly imported')
 report={'format':'independent-completion-v3-next-objective-design-review-v1','complete':True,'independent':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(Path(__file__)),'sources':sources,'saved_recomputation':{'paired_endpoint_arrays':96,'HQ_measurements_per_array':4,'maximum_HQ_receipt_deviation':maxdev,'all_delivered_endpoint_receipt_fields_crosschecked':True,'all_aggregate_arithmetic_recomputed':True,'all_HQ_plus_boundary_group_totals_improve':True,'individual_HQ_hole_improves_PNG_hole_worsens_cases':0,'group_gains_percent':recomputed},'recommendation':{'design_admissible_for_preparation':True,'route':'App-delivered clamped bilinear256 masked RGB hole-L1 only','start':'Exact full R1 parent, not failed V3 or incomplete V2','frozen':443,'trainable_existing_fusion':54,'schedule':'Same256cases/64cycles/4microbatches','code_Adam':'Restored unchanged at32; zero new steps','fusion_Adam':'Restored32;64finite new steps to96; retain LR1e-5 and current clip1','training_objective':'Mean abs(delivered_float256-clean256) over fixedmask RGB pixels; differentiable tensor clamp->bilinear512to256->exact source composite; microbatchmean/4. No added seam term, no PNG rounding in autograd.','all_quality_and_preservation_gates_unchanged':True,'automatic_launch_or_resume':False,'model_qualified':False},'alternatives':{'HQ512_hole_only':'Cleaner single removal of boundary compensation, but leaves measured target/resolution/output mismatch. Useful causal alternative, lower priority for the256delivery target.','app256_hole_only':'Preferred practical alignment: removes boundary compensation and uses actual float delivery resolution/targets; changes two objective aspects, so not a solecause experiment.','reweighting_HQ512_hole_boundary':'Can reduce compensation but weight is unestablished and cannot enforce each subgroup/gate; adaptive tuning risks chasing reused monitor.','train_longer_or_increase_LR':'Not justified by this return: intended sum improves while some holes worsen, and no clip bottleneck was observed. More steps can continue the wrong tradeoff.'},'required_preparation_checks':['New immutable namespace; preserve V2/V3 failures and exact R1 baseline.','Before any update reproduce all4fresh app parity and48R1 raw<=1e-6/PNGexact cases.','Loss tensor must use clean256 and exact current clamp-before-bilinear order, fixed support, no detach/numpy/rint/PNG inside gradient path.','No outside or empty-mask gradient/optimizer updates; retain exact visible256 and empty bypass output guarantees.','Keep separate>=1percent clear/challenge PNG holegain, all raw/PNG macro/geometry/condition hole+boundary nonworse1e-12 and10percent catastrophic stops.','Retain54finite fusion gradient receipts,443frozen and codeAdam bitexact guards, full named Adam/RNG/sampler/partial failure state, phase timing/resource/export caps.','Record clamp saturation; do not introduce straight-through clipping or change gates silently.','Independent saved-output/state audits and all planned visuals/user review before adoption.'], 'limitations':['This saved-only monitor/challenge diagnosis does not show matched before/after256FIT training losses or establish a sole cause.','The clear lens HQ/PNG sign flip is aggregate only; no individual terminal case has that sign flip.','Tensor clamp has zero gradient outside its bounds, and delivered PNG quantization remains non-differentiable; float256 is a closer surrogate, not the PNG endpoint itself.','Hole-only may worsen seams; unchanged boundary gates must reject that outcome.','Masked RGB L1 can prefer averages/softness; it does not establish plausible hidden anatomy, identity recovery or native CCTV quality.','Adaptive TRAIN monitoring is not independent FINAL; root20sheet review found no consistent major V3 benefit.','No new model/pilot packet/training/app changes are authorized by this review alone.'],'local_execution':{'Torch_imported':False,'neural_calls':0,'gradient_calls':0,'optimizer_operations':0,'VM_calls':0,'returned_outputs_edited':False},'input_arrays':input_pins,'checks_passed':CHECKS,'seconds':time.monotonic()-started,'model_qualified':False,'goal_complete':False}
 OUT.mkdir();target=OUT/'independent_design_review.json';target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 print(json.dumps({'complete':True,'review':pin(target),'checks':CHECKS,'endpoint_arrays':96,'max_HQ_deviation':maxdev,'recommendation':report['recommendation']['route'],'training_or_packet_created':False,'seconds':report['seconds']},indent=2))
if __name__=='__main__':main()

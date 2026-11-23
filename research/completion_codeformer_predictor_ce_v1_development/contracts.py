"""Prospective finite predictor-only CE contract, not a release."""
from pathlib import Path
import hashlib,json
VM_NAME='cctv_completion_codeformer_predictor_ce_v1_vm'
ORIGINAL_SHA='b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797'
PARENT_SHA='0a6b4ad907c6ead86d8f26d3446666387493cdce81c6adc1ed78a8677d50fadd'
PARENT_PROTOCOL_SHA='8815ad168ab429bf4f30b8d719d3a6de6d1e2667fc38412c89c9c95e74e5ff3d'
PARENT_LOGICAL='55feaa296f5164a6bd12995b487c9366c5fe7778f0e9dbe8912fc4ee7c4122d6'
ORIGINAL_LOGICAL='586a8854ed7b5e3f0e202c258addb7a87de63cd3068d153741b065379ff6668f'
SUBSET_SHA='8ec2d535cd6fb231af815d0ec833a1cd08f418458cbc4919c0ebd13f638f2a1d'
KINDS=('lower_face','eye_band','image_left_lens','cheek_object')
METRICS=('raw_hole_L1','raw_boundary_L1','PNG_hole_L1','PNG_boundary_L1')
OBJECTIVE_ID='predictor_only_CE_all256_same_inpainting_codebook_v1'
PREDICTOR_PREFIXES=('feat_emb.','ft_layers.','idx_pred_layer.')
TYPED_CAPS={'parity':8,'fit_code':128,'monitor':128,'challenge':64,'public_development':56}
CONTRACT={'cycles':32,'microbatches_per_route':4,'batch_size':1,'optimizer_steps_per_cycle':1,
 'optimizer_steps_max':32,'backward_calls_max':128,'forward_slots_max':420,'forward_slots_planned':384,
 'whole_model_invocations_planned':384,'generator_model_forwards_planned':256,'code_only_forwards_planned':128,
 'teacher_target_sets':80,'teacher_component_calls_planned':0,'teacher_calls_max':0,
 'fit_references':32,'monitor_references':8,'fit_cases':128,'fit_full_pool_cases':256,
 'fit_anchor_cases':64,'fit_augmented_cases':64,'monitor_cases':32,'challenge_cases':16,
 'development_cases':32,'snapshots':[0,8,16,32],'challenge_snapshots':[0,8,16,32],
 'public_snapshots':[0,32],'worker_seconds':1800,'cooperative_stop_seconds':1680,
 'save_reserve_seconds':120,'export_cooperative_seconds':900,'output_bytes':2560*1024**2,
 'minimum_free_bytes':8*1024**3,'stop_free_bytes':2*1024**3,'CUDA_allocated_bytes_cap':20*1024**3,
 'terminal_bytes':1024**3,'teacher_bytes':24*1024**2,'metadata_bytes':8*1024**2,
 'trainer_log_bytes':4*1024**2,'PNG_bytes_each':256*1024,'precision':'FP32_no_autocast_no_TF32',
 'codebook_size':512,'fusion_sizes':['32','64','128'],'fusion_w':1,
 'Adam':{'lr_predictor':1e-5,'betas':[.9,.99],'eps':1e-8,'weight_decay':0.,'clip_L2':1.},
 'trainable_predictor_state_tensors':114,'frozen_state_tensors':383,
 'code_loss':'mean_CE_all256_same_codebook_indices_divided_once_by4',
 'latent_MSE_added':False,'fusion_training':False,'encoder_training':False,'oracle_inference':False,
 'selection_min_macro_PNG_hole_gain':.01,'nonworse_tolerance':1e-12,'catastrophic_regression_fraction':.10,
 'parent_completed_cycles':32,'parent_optimizer_steps_per_route':32,'parent_backward_calls':256,
 'parent_predictor_moments_mapped_by_name':True,'parent_inactive_encoder_slots_preserved':164,
 'parent_inactive_fusion_slots_preserved':54,'terminal_mixed_code_slot_steps_explicit':True,
 'saved_FIT_logits_npz':128,'saved_eval_npz':256,'saved_parity_npz':4,'saved_NPZ_max':388,
 'empty_bypasses_planned':168,'GRAY_style_gate':{'cases':8,'macro_and_four_geometry_nonworse':True,'raw_tolerance':1e-12,'PNG_tolerance':1e-12,'only_final32_required':True},'original_VAL_used':False,'reserved_FINAL_used':False,
 'automatic_launch':False,'automatic_resume':False}
def predictor_name(name):return name=='position_emb' or name.startswith(PREDICTOR_PREFIXES)
def need(ok,message):
 if not ok:raise ValueError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(path):
 def reject(x):raise ValueError('Nonfinite JSON '+x)
 return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=reject)
def aggregate(rows,condition=False):
 need(rows,'Cannot aggregate empty evaluated cohort')
 def mean(part):return {k:sum(r[k] for r in part)/len(part) for k in METRICS}
 result={'macro':mean(rows),'geometry':{kind:mean([r for r in rows if r['mask_kind']==kind]) for kind in KINDS}}
 if condition:
  result['condition']={name:mean([r for r in rows if r['condition']==name]) for name in sorted({r['condition'] for r in rows})}
 return result
def compare(now,baseline,*,conditions=False):
 a,b=now['metrics'],baseline['metrics'];groups=['macro','geometry']+(['condition'] if conditions else [])
 pairs=[(a['macro'],b['macro'])]
 for group in groups[1:]:
  need(set(a[group])==set(b[group]),'Compared subgroup membership differs')
  pairs.extend((a[group][key],b[group][key]) for key in a[group])
 nonworse=all(x[key]<=y[key]+1e-12 for x,y in pairs for key in METRICS)
 catastrophic=any(x[key]>y[key]*1.10 for x,y in pairs for key in METRICS)
 gain=(b['macro']['PNG_hole_L1']-a['macro']['PNG_hole_L1'])/max(b['macro']['PNG_hole_L1'],1e-12)
 return {'prospective_numeric_screen_pass':bool(gain>=.01 and nonworse),'PNG_macro_hole_gain':gain,
  'all_raw_PNG_hole_boundary_groups_nonworse':bool(nonworse),'catastrophic_regression_stop':bool(catastrophic),
  'nonworse_tolerance':1e-12,'condition_groups_included':conditions,'visual_review_pending':True,'model_qualified':False}


def gray_style(summary):
 rows=[r for r in summary['records'] if r['style']=='GRAY']
 need(len(rows)==8 and len({r['id'] for r in rows})==8,'Exactly8fixed GRAY challenge cases')
 def avg(part):
  need(bool(part),'GRAY subgroup missing')
  return {k:sum(r[k] for r in part)/len(part) for k in ('GRAY_raw_mean_channel_spread','GRAY_PNG_mean_channel_spread')}
 geometry={kind:[r for r in rows if r['mask_kind']==kind] for kind in KINDS}
 need(all(len(v)==2 for v in geometry.values()),'Exactly2GRAYcases per fixed geometry')
 return {'macro':avg(rows),'geometry':{k:avg(v) for k,v in geometry.items()},'cases':[{k:r[k] for k in ('id','mask_kind','GRAY_raw_mean_channel_spread','GRAY_PNG_mean_channel_spread')} for r in rows]}
def compare_gray(now,baseline):
 a,b=gray_style(now),gray_style(baseline);pairs=[('macro',a['macro'],b['macro'])]+[('geometry/'+k,a['geometry'][k],b['geometry'][k]) for k in KINDS]
 failures=[{'group':label,'metric':k,'candidate':x[k],'baseline':y[k],'delta':x[k]-y[k]} for label,x,y in pairs for k in x if x[k]>y[k]+1e-12]
 return {'all_GRAY_macro_geometry_chroma_nonworse':not failures,'nonworse_failures':failures,'tolerance_raw':1e-12,'tolerance_PNG':1e-12,'candidate':a,'R1_baseline':b,'model_qualified':False}

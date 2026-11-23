"""Pure finite SAME-codebook zero-optimizer probe contract."""
from pathlib import Path
import hashlib,json
VM_NAME='cctv_completion_codeformer_spatial_code_intervention_v1_vm'
ORIGINAL_SHA='b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797'
PARENT_SHA='0a6b4ad907c6ead86d8f26d3446666387493cdce81c6adc1ed78a8677d50fadd'
PARENT_PROTOCOL_SHA='8815ad168ab429bf4f30b8d719d3a6de6d1e2667fc38412c89c9c95e74e5ff3d'
PARENT_LOGICAL='55feaa296f5164a6bd12995b487c9366c5fe7778f0e9dbe8912fc4ee7c4122d6'
ORIGINAL_LOGICAL='586a8854ed7b5e3f0e202c258addb7a87de63cd3068d153741b065379ff6668f'
KINDS=('lower_face','eye_band','image_left_lens','cheek_object')
METRICS=('raw_hole_L1','raw_boundary_L1','PNG_hole_L1','PNG_boundary_L1')
ARMS=[{'id':'H','w':1,'oracle':True,'selection':'touched','filename_tag':'ORACLE_H_mask_touched_same_codebook'},
      {'id':'V','w':1,'oracle':True,'selection':'visible','filename_tag':'ORACLE_V_visible_context_same_codebook'}]
EXTERNAL_ARMS={'A':'ACTUAL_A_R1_w1','C':'ORACLE_C_unavailable_clean_information_w1'}
TYPED_CAPS={'parity':8,'bridge':4,'H':48,'V':48}
PROBE_PROTOCOL_SHA='51782b57fca9ad9cf7161b0f4451a65e545e3eb2f8f58788fa40b64aec48d1ec'
CONTRACT={'cases':48,'monitor_cases':32,'challenge_cases':16,'arms':2,'outputs':96,
 'parity_pairs':4,'bridge_cases':4,'forward_slots_planned':108,'forward_slots_max':120,
 'typed_forward_caps':TYPED_CAPS,'gradient_calls':0,'backward_calls_max':0,'optimizer_steps_max':0,
 'teacher_component_calls_planned':0,'worker_seconds':900,'cooperative_stop_seconds':840,'failure_save_reserve_seconds':60,
 'output_bytes':1024**3,'minimum_free_bytes':6*1024**3,'stop_free_bytes':2*1024**3,
 'CUDA_allocated_bytes_cap':20*1024**3,'terminal_bytes':384*1024**2,'teacher_bytes':0,
 'array_bytes':560*1024**2,'PNG_bytes_total':24*1024**2,'metadata_bytes':8*1024**2,
 'trainer_log_bytes':4*1024**2,'PNG_bytes_each':256*1024,'precision':'FP32_no_autocast_no_TF32',
 'state_tensors_frozen':497,'codebook_size':512,'oracle_label_sets':16,
 'grid_rule':'16x16 original256 cell touched iff any reviewed binary removal pixel; fixed no dilation/search',
 'new_NPZ_max':104,'new_PNG_max':96,'external_control_files':196,
 'all_outputs_diagnostic_only':True,'automatic_resume':False,'automatic_launch':False,
 'optimizer_objects':0,'checkpoint_save':False,'unexpected_mutation_failure_state_only':True,
 'failure_state_bytes':384*1024**2,'return_raw_tar_and_gzip_bytes':1280*1024**2,
 'export_seconds':900,'external_control_recompute':False}

def output_paths(protocol):
 top=['state_before.json','oracle_label_receipt.json','reused_control_binding.json','external_control_compatibility.json',
      'arm_H_complete.json','arm_V_complete.json','state_after.json','results.json','failure.json',
      'state_on_failure.json','state_receipt_failure.json','failure_state_receipt.json',
      'failed_actual_model_state.pth','failed_actual_model_state.pth.partial']
 paths=['outputs/'+name for name in top]
 for folder,key in [('fresh_app_parity','parity_cases'),('external_bridge','bridge_cases')]:
  paths.append('outputs/'+folder+'/summary.json')
  for case in protocol[key]:
   for suffix in ('.npz','.json'):paths.append('outputs/'+folder+'/'+case['id']+suffix)
 for arm in ARMS:
  for kind,key in [('monitor','monitor_cases'),('challenge','challenge_cases')]:
   base='outputs/'+arm['filename_tag']+'/'+kind+'/'
   paths.append(base+'results.json')
   for case in protocol[key]:
    for suffix in ('.npz','.png'):paths.append(base+case['id']+'__ORACLE_unavailable_clean_information'+suffix)
 return sorted(paths)


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


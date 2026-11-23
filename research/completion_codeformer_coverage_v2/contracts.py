"""Pure contracts and bounded output writers for one manual coverage-v2 pilot."""
from pathlib import Path
import hashlib,json,os,shutil,time
VM_NAME='cctv_completion_codeformer_coverage_v2_vm'
ORIGINAL_SHA='b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797'
PARENT_SHA='0a6b4ad907c6ead86d8f26d3446666387493cdce81c6adc1ed78a8677d50fadd'
PARENT_PROTOCOL_SHA='8815ad168ab429bf4f30b8d719d3a6de6d1e2667fc38412c89c9c95e74e5ff3d'
PARENT_LOGICAL='55feaa296f5164a6bd12995b487c9366c5fe7778f0e9dbe8912fc4ee7c4122d6'
ORIGINAL_LOGICAL='586a8854ed7b5e3f0e202c258addb7a87de63cd3068d153741b065379ff6668f'
KINDS=('lower_face','eye_band','image_left_lens','cheek_object')
METRICS=('raw_hole_L1','raw_boundary_L1','PNG_hole_L1','PNG_boundary_L1')
CONTRACT={'cycles':64,'microbatches_per_route':4,'batch_size':1,'optimizer_steps_per_cycle':2,
 'optimizer_steps_max':128,'backward_calls_max':512,'forward_slots_max':1000,'forward_slots_planned':896,
 'whole_model_forwards_planned':736,'teacher_target_sets':80,'teacher_component_calls_planned':160,
 'fit_references':32,'monitor_references':8,'fit_cases':256,'fit_anchor_cases':128,'fit_augmented_cases':128,
 'monitor_cases':32,'challenge_cases':16,'development_cases':32,'snapshots':[0,16,32,64],
 'challenge_snapshots':[0,64],'public_snapshots':[0,64],'worker_seconds':1800,'cooperative_stop_seconds':1680,
 'output_bytes':2304*1024**2,'minimum_free_bytes':7*1024**3,'stop_free_bytes':2*1024**3,
 'CUDA_allocated_bytes_cap':20*1024**3,'terminal_bytes':1024**3,'teacher_bytes':24*1024**2,
 'metadata_bytes':8*1024**2,'trainer_log_bytes':4*1024**2,'PNG_bytes_each':256*1024,
 'precision':'FP32_no_autocast_no_TF32','codebook_size':512,'fusion_sizes':['32','64','128'],
 'Adam':{'lr_code':1e-5,'lr_fusion':1e-5,'betas':[.9,.99],'eps':1e-8,'weight_decay':0.,'clip_L2':1.},
 'code_loss':'CE_all256_clean_nearest_codes + MSE_clean_HQ512_encoder_latent',
 'fusion_loss':'unclamped_mask_RGB512_L1 + inner_boundary_RGB512_L1_kernel7',
 'selection_min_macro_PNG_hole_gain':.01,'nonworse_tolerance':1e-12,'catastrophic_regression_fraction':.10,
 'parent_completed_cycles':32,'parent_optimizer_steps_per_route':32,'parent_backward_calls':256,
 'seed_for_initialization':240,'teacher_limitation':'same original frozen inpainting encoder/codebook, not independent clean prior',
 'original_VAL_used':False,'reserved_FINAL_used':False,'automatic_launch':False,'automatic_resume':False}

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

class OutputBudget:
 """Track every owned write, reserve a complete failed terminal and small receipt.

 Limits are on file extents, so ZIP header seeks do not double count. Parent
 logs are external and sampled separately; the supervisor streams under4MiB.
 """
 def __init__(self,out,started,*,disk_free=None,clock=None):
  self.out=Path(out).resolve();self.started=started;self.clock=clock or time.monotonic
  self.disk_free=disk_free or (lambda:shutil.disk_usage(self.out).free);self.sizes={};self.metadata_external=[];self.terminal_committed=False;self.supervisor_reserve=1024**2
 def category(self,path):
  p=Path(path).resolve();need(p.is_relative_to(self.out),'Output writer escaped evidence root')
  rel=p.relative_to(self.out).as_posix()
  if rel in ('terminal_training_state.pth','terminal_training_state.pth.partial'):return 'terminal'
  if rel.startswith('teacher_labels/') and rel.endswith('.npz'):return 'teacher'
  if rel.endswith('.png'):return 'PNG'
  if rel.endswith(('.json','.log')):return 'metadata'
  need(rel.endswith('.npz'),'Unexpected output evidence type');return 'array'
 def total(self,category=None):
  return sum(size for path,size in self.sizes.items() if category is None or self.category(path)==category)
 def external_metadata(self):
  return sum(p.stat().st_size for p in self.metadata_external if p.is_file())
 def guard(self,*,terminal=False,emergency=False):
  limit=1800 if terminal or emergency or self.terminal_committed else 1680
  need(self.clock()-self.started<limit,'Finite worker deadline; preserve terminal state')
  need(self.total()+self.external_metadata()+self.supervisor_reserve<=CONTRACT['output_bytes'],'Evidence/export metadata cap')
  need(self.total('teacher')<=CONTRACT['teacher_bytes'],'Teacher label cap')
  need(self.total('terminal')<=CONTRACT['terminal_bytes'],'Terminal serialization cap')
  need(self.total('metadata')+self.external_metadata()+self.supervisor_reserve<=CONTRACT['metadata_bytes'],'Metadata/log/supervisor cap')
  reserve=(0 if terminal or emergency else 1024**2 if self.terminal_committed else CONTRACT['terminal_bytes']+1024**2)
  need(self.disk_free()>=CONTRACT['stop_free_bytes']+reserve,'Stop before reserved terminal/free disk is consumed')
 def before_write(self,path,end,*,terminal=False,emergency=False):
  self.guard(terminal=terminal,emergency=emergency);p=Path(path).resolve();category=self.category(p)
  old=self.sizes.get(p,0);growth=max(0,end-old)
  need(self.total()+self.external_metadata()+self.supervisor_reserve+growth<=CONTRACT['output_bytes'],'Whole output cap before write')
  if category=='terminal':need(self.total('terminal')+growth<=CONTRACT['terminal_bytes'],'Terminal write cap')
  elif category=='teacher':need(self.total('teacher')+growth<=CONTRACT['teacher_bytes'],'Retained teacher cap')
  elif category=='metadata':
   metadata_cap=CONTRACT['metadata_bytes']-self.supervisor_reserve-(0 if emergency else 512*1024)
   need(self.total('metadata')+self.external_metadata()+growth<=metadata_cap,'Metadata write cap with failure receipt reserve')
  elif category=='PNG':need(end<=CONTRACT['PNG_bytes_each'],'PNG extent cap')
  if category!='terminal' and not emergency and not self.terminal_committed:
   need(self.total()+self.external_metadata()+self.supervisor_reserve+growth<=CONTRACT['output_bytes']-CONTRACT['terminal_bytes']-1024**2,'Preserve terminal byte reservation')
  if self.terminal_committed and not terminal and not emergency:
   need(self.total()+self.external_metadata()+self.supervisor_reserve+growth<=CONTRACT['output_bytes']-1024**2,'Preserve final failure receipt headroom')
  need(self.disk_free()-growth>=CONTRACT['stop_free_bytes'],'Preserve2GiBfree while writing')
 def rename(self,before,after):
  self.sizes[Path(after).resolve()]=self.sizes.pop(Path(before).resolve())
  if Path(after).name=='terminal_training_state.pth':self.terminal_committed=True
 def open(self,path,*,terminal=False,emergency=False):return CappedWriter(self,path,terminal=terminal,emergency=emergency)
 def json(self,path,value,*,emergency=False):
  body=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode('utf-8')
  with self.open(path,emergency=emergency) as stream:stream.write(body)
class CappedWriter:
 def __init__(self,budget,path,*,terminal=False,emergency=False):
  self.budget=budget;self.path=Path(path);self.terminal=terminal;self.emergency=emergency
  self.path.parent.mkdir(parents=True,exist_ok=True);need(not self.path.is_symlink(),'Linked evidence path refused')
  self.stream=self.path.open('xb');self.budget.sizes[self.path.resolve()]=0
 def write(self,data):
  end=self.stream.tell()+len(data);self.budget.before_write(self.path,end,terminal=self.terminal,emergency=self.emergency)
  n=self.stream.write(data);need(n==len(data),'Short evidence write')
  self.budget.sizes[self.path.resolve()]=max(self.budget.sizes[self.path.resolve()],end);return n
 def read(self,*args):return self.stream.read(*args)
 def tell(self):return self.stream.tell()
 def seek(self,*args):return self.stream.seek(*args)
 def flush(self):return self.stream.flush()
 def writable(self):return True
 def seekable(self):return True
 @property
 def closed(self):return self.stream.closed
 def close(self):return self.stream.close()
 def __enter__(self):return self
 def __exit__(self,*args):self.close()

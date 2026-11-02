"""Exact CPU tensor clone boundary test; no module, optimizer, autograd or VM."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,importlib.util,json,sys
import torch
HERE=Path(__file__).resolve().parent
worker=HERE/'cctv_completion_codeformer_predictor_ce_v1_vm.py';source=worker.read_text();tree=ast.parse(source);func=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='cpu_copy')
module=ast.Module(body=[func],type_ignores=[]);scope={'torch':torch};exec(compile(ast.fix_missing_locations(module),'<exact-worker-cpu-copy>','exec'),scope);clone=scope['cpu_copy']
spec=importlib.util.spec_from_file_location('pure_topology',HERE/'optimizer_topology.py');top=importlib.util.module_from_spec(spec);spec.loader.exec_module(top)
checks=0
def need(ok,label):
 global checks
 checks+=1
 if not ok:raise AssertionError(label)
original_names=['encoder.'+str(i) for i in range(164)]+['predictor.'+str(i) for i in range(114)];ids=[13+7*i for i in range(278)]
parent={'state':{ident:{'step':torch.tensor(32.,dtype=torch.float32),'exp_avg':torch.tensor([float(i),-.5],dtype=torch.float32),'exp_avg_sq':torch.tensor([float(i)+1,.25],dtype=torch.float32)} for i,ident in enumerate(ids)},'param_groups':[{'params':ids,'lr':1e-5,'betas':(.9,.99),'eps':1e-8,'weight_decay':0.}]}
before=clone(parent);active_names=original_names[164:];projected=top.project(parent,original_names,active_names,clone=clone)
for i,name in enumerate(active_names):
 old=parent['state'][ids[164+i]];new=projected['state'][i]
 for key in ('step','exp_avg','exp_avg_sq'):
  need(torch.equal(new[key],old[key]),'Copied CPU value exact '+name+key);need(new[key].data_ptr()!=old[key].data_ptr(),'Projected tensor must own independent storage '+name+key);new[key].add_(1)
for ident in ids:
 for key in ('step','exp_avg','exp_avg_sq'):need(torch.equal(parent['state'][ident][key],before['state'][ident][key]),'Projected writes cannot mutate retained parent')
merged=top.merge(parent,original_names,projected,active_names,clone=clone)
for i,ident in enumerate(ids):
 for key in ('step','exp_avg','exp_avg_sq'):
  expected=before['state'][ident][key] if i<164 else projected['state'][i-164][key]
  need(torch.equal(merged['state'][ident][key],expected),'Merged full278 slots exact');need(merged['state'][ident][key].data_ptr()!=parent['state'][ident][key].data_ptr(),'Merged full slots independent')
need(set(merged)=={'state','param_groups'} and len(merged['state'])==278,'Exactlyone full mixed278 state without duplicated114 optimizer')
need(not any(t.requires_grad for state in parent['state'].values() for t in state.values()),'All plain tensors have no autograd')
# Execute the actual per-parameter gradient receipt on plain CPU tensor attributes.
import math
from types import SimpleNamespace
worker_fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker');grad_fn=next(n for n in worker_fn.body if isinstance(n,ast.FunctionDef) and n.name=='gradient_receipt')
parameters=[SimpleNamespace(grad=torch.zeros(2) if i==0 else torch.ones(2)) for i in range(114)]
grad_scope={'torch':torch,'math':math,'need':need,'predictor_names':active_names,'predictor_parameters':parameters}
exec(compile(ast.fix_missing_locations(ast.Module(body=[grad_fn],type_ignores=[])),'<exact-worker-gradient-receipt>','exec'),grad_scope)
receipt=grad_scope['gradient_receipt']({'fixed_group':parameters});need(len(receipt['per_parameter'])==114,'All114 per-name actualgradient entries');need(receipt['per_parameter'][active_names[0]]['zero_retained_honestly'] and not receipt['per_parameter'][active_names[0]]['nonzero'],'Actualzero norm retained without pretend nonzero');need(all(receipt['per_parameter'][n]['finite'] for n in active_names),'AllplainCPUgradientattributesfinite')
# Check exact final named114 moment validation with plain tensors and a state container, not an optimizer.
validate_fn=next(n for n in worker_fn.body if isinstance(n,ast.FunctionDef) and n.name=='validate_final_predictor_Adam')
plain_parameters=[torch.zeros(2,dtype=torch.float32) for _ in range(114)]
plain_state={parameter:{'step':torch.tensor(64.,dtype=torch.float32),'exp_avg':torch.ones_like(parameter),'exp_avg_sq':torch.ones_like(parameter)} for parameter in plain_parameters}
validation_scope={'torch':torch,'need':need,'predictor_optimizer':SimpleNamespace(state=plain_state),'predictor_names':active_names,'predictor_parameters':plain_parameters,'counts':{'predictor_optimizer_steps':32}}
exec(compile(ast.fix_missing_locations(ast.Module(body=[validate_fn],type_ignores=[])),'<exact-final-Adam-validation>','exec'),validation_scope)
need(validation_scope['validate_final_predictor_Adam'](),'Exact finite named114 final slots accepted')
first=plain_state[plain_parameters[0]]
for key,bad in [('step',torch.tensor(33.)),('step',torch.tensor([64.])),('exp_avg',torch.tensor([float('nan'),1.])),('exp_avg_sq',torch.tensor([-1.,1.]))]:
 old=first[key];first[key]=bad;rejected=False
 try:validation_scope['validate_final_predictor_Adam']()
 except AssertionError:rejected=True
 finally:first[key]=old
 need(rejected,'Malformed/nonfinite/negative final Adam slot refused')


report={'format':'predictor-ce-v1-exact-worker-CPUclone-alias-fixture-v1','complete':True,'checks':checks,'UTC':datetime.now(timezone.utc).isoformat(),'exact_cpu_copy_source':{'path':str(worker),'bytes':worker.stat().st_size,'sha256':hashlib.sha256(worker.read_bytes()).hexdigest()},'original_slots':278,'projected_slots':114,'active_step_and_moment_aliases':False,'mutating_projected_step_and_moments_preserves_parent':True,'merge_retains_original164_inactive_slots_and_updated114_once':True,'execution':{'Torch_CPU_tensor_operations_only':True,'model_or_Module_objects':0,'optimizer_objects':0,'gradient_queries':0,'autograd_calls':0,'training_steps':0,'neural_calls':0,'VM_calls':0}}
target=HERE/'cpu_clone_alias_fixture.json';target.write_bytes((json.dumps(report,indent=2)+'\n').encode());print(json.dumps(report,indent=2))

"""Independent source/plain CPU tensor semantics. No neural model/optimizer objects or training."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,importlib.util,json,math,sys,tempfile,time,types
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'research/completion_codeformer_predictor_ce_v1_development'
checks=[]
def need(ok,label):
 if not ok:raise ValueError(label)
 checks.append(label)
def pin(p):p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def extract(fn,scope):
 exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<exact-worker-'+fn.name+'>','exec'),scope);return scope[fn.name]
def main():
 start=time.monotonic();source=HERE/'cctv_completion_codeformer_predictor_ce_v1_vm.py';initial_source_bytes=source.read_bytes();tree=ast.parse(source.read_text());worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker');nested={n.name:n for n in worker.body if isinstance(n,ast.FunctionDef)}
 for f in HERE.glob('*.py'):ast.parse(f.read_text(),feature_version=(3,10));need(True,'Python3.10 syntax '+f.name)
 for n in tree.body:
  if isinstance(n,(ast.Import,ast.ImportFrom)):need(not (isinstance(n,ast.ImportFrom) and n.module and n.module.startswith(('torch','numpy','PIL'))) and not (isinstance(n,ast.Import) and any(v.name.startswith(('torch','numpy','PIL')) for v in n.names)),'No scientific import before guarded worker')
 adam=[n for n in ast.walk(worker) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Adam'];need(len(adam)==1,'One active Adam object only')
 optimizer_calls=[ast.unparse(n.func) for n in ast.walk(worker) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('step','zero_grad')];need(set(optimizer_calls)=={'predictor_optimizer.step','predictor_optimizer.zero_grad'},'Only114 active optimizer can step or zero')
 net_calls=[n for n in ast.walk(worker) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='net'];need(any(any(k.arg=='code_only' and isinstance(k.value,ast.Constant) and k.value.value is True for k in n.keywords) for n in net_calls),'FIT uses graph-connected code_only logits')
 need(not any(any(k.arg=='oracle_indices' for k in n.keywords) for n in net_calls),'No oracle override in deployment or FIT')
 arch=ROOT/'third_party/codeformer/codeformer_arch.py';at=ast.parse(arch.read_text());forward=next(n for n in ast.walk(at) if isinstance(n,ast.FunctionDef) and n.name=='forward' and any(a.arg=='code_only' for a in n.args.args));code_if=next(n for n in forward.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='code_only');need(any(isinstance(n,ast.Return) and ast.unparse(n.value)=='(logits, lq_feat)' for n in code_if.body),'Local code_only returns logits before quantization')
 code_index=forward.body.index(code_if);need(not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('topk','get_codebook_feat','detach') for node in forward.body[:code_index] for n in ast.walk(node)),'No hard-top1, codebook lookup or detach before CE return')
 need(ast.unparse(nested['active_predictor']).count('requires_grad_(True)')==1 and 'predictor_names' in ast.unparse(nested['active_predictor']),'Only exact predictor allowlist enabled')
 contract=load('research.completion_codeformer_predictor_ce_v1.contracts',HERE/'contracts.py');top=load('independent_predictor_topology',HERE/'optimizer_topology.py')
 import torch
 torch.set_num_threads(1)
 cpu_copy=extract(nested['cpu_copy'],{'torch':torch});parent_names=['encoder.'+str(i) for i in range(164)]+['position_emb']+['ft_layers.'+str(i) for i in range(113)];active_names=parent_names[164:];ids=[1000+i*3 for i in range(278)]
 original={'state':{ident:{'step':torch.tensor(32.,dtype=torch.float32),'exp_avg':torch.tensor([i+.25,i+.5],dtype=torch.float32),'exp_avg_sq':torch.tensor([i+.75,i+1.],dtype=torch.float32)} for i,ident in enumerate(ids)},'param_groups':[{'params':ids,'lr':1e-5,'betas':(.9,.99),'eps':1e-8,'weight_decay':0.,'capturable':False}]}
 retained=cpu_copy(original);projected=top.project(retained,parent_names,active_names,clone=cpu_copy)
 for i,name in enumerate(active_names):
  old=retained['state'][ids[164+i]];new=projected['state'][i]
  for k in old:
   need(torch.equal(old[k],new[k]) and old[k].data_ptr()!=new[k].data_ptr(),'Mapped exact detached storage '+name+'/'+k)
  new['step'].add_(32);new['exp_avg'].add_(.125);new['exp_avg_sq'].add_(.25)
 merged=top.merge(retained,parent_names,projected,active_names,clone=cpu_copy)
 for i,ident in enumerate(ids):
  need(float(retained['state'][ident]['step'])==32 and all(torch.equal(retained['state'][ident][k],original['state'][ident][k]) for k in original['state'][ident]),'Parent immutable '+parent_names[i])
  need(float(merged['state'][ident]['step'])==(32 if i<164 else 64),'Mixed exact step '+parent_names[i])
  for k,v in merged['state'][ident].items():need(v.data_ptr()!=retained['state'][ident][k].data_ptr(),'Merged storage independent '+parent_names[i]+'/'+k)
 need(merged['param_groups']==original['param_groups'] and len(merged['state'])==278,'Full278 historical metadata preserved without duplicate114 payload')
 for names in [active_names+[active_names[0]],['absent']]:
  try:top.project(retained,parent_names,names,clone=cpu_copy)
  except ValueError:need(True,'Invalid active topology refuses')
  else:raise ValueError('Invalid topology admitted')
 # Exact nested gradient reporting on plain CPU tensors; no Parameters/modules.
 tensors=[torch.tensor([float(i),.25]) for i in range(114)]
 for i,t in enumerate(tensors):t.grad=torch.zeros_like(t) if i==0 else torch.tensor([.01,.02])
 gradient=extract(nested['gradient_receipt'],{'torch':torch,'math':math,'need':need,'predictor_names':active_names,'predictor_parameters':tensors})({'all':tensors})
 need(len(gradient['per_parameter'])==114 and gradient['per_parameter'][active_names[0]]['zero_retained_honestly'] and not gradient['per_parameter'][active_names[0]]['nonzero'],'All114 gradients reported with honest zero')
 # Exact CE/loss assignment from the worker, using only a leaf logits tensor.
 ce_assign=next(n for n in ast.walk(worker) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ce' for t in n.targets));loss_assign=next(n for n in ast.walk(worker) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='loss' for t in n.targets));need(ast.unparse(ce_assign)=="ce = F.cross_entropy(logits.reshape(-1, 512), labels.reshape(-1), reduction='mean')" and ast.unparse(loss_assign)=='loss = ce / 4','Exact all256 mean CE divided once by4')
 import torch.nn.functional as F
 rng=np.random.default_rng(42);ce_checks=[]
 for micro in range(4):
  a=rng.normal(0,.5,(1,256,512)).astype(np.float32);y=rng.integers(0,512,(1,256),dtype=np.int64);logits=torch.tensor(a,requires_grad=True);labels=torch.from_numpy(y);scope={'F':F,'logits':logits,'labels':labels}
  exec(compile(ast.fix_missing_locations(ast.Module(body=[ce_assign,loss_assign],type_ignores=[])),'<exact-CE-loss>','exec'),scope);g=torch.autograd.grad(scope['loss'],logits)[0].detach().numpy()
  d=a.astype(np.float64);d-=d.max(-1,keepdims=True);probs=np.exp(d);probs/=probs.sum(-1,keepdims=True);expected=probs.copy();np.put_along_axis(expected,y[...,None],np.take_along_axis(expected,y[...,None],-1)-1,axis=-1);expected/=256*4
  ce=float(np.mean(np.log(np.exp(d).sum(-1))-np.take_along_axis(d,y[...,None],axis=-1)[...,0]));need(abs(scope['ce'].item()-ce)<1e-6,'CE forward analytical agreement '+str(micro));need(np.max(np.abs(g-expected))<2e-10,'CE gradient analytic /256/4 '+str(micro));ce_checks.append({'microbatch':micro,'forward_absolute_error':abs(scope['ce'].item()-ce),'maximum_gradient_error':float(np.max(np.abs(g-expected)))})
 # Endpoint gate is exact extracted expression; earlier diagnostic pass cannot qualify.
 fs=next(n for n in ast.walk(worker) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='final_screen' for t in n.targets))
 for cycle in (8,16,32):
  for stop in (None,'catastrophic stop'):
   scope={'terminal_cycle':cycle,'stop_reason':stop,'clear_screen':{'prospective_numeric_screen_pass':True},'challenge_screen':{'prospective_numeric_screen_pass':True},'gray_screen':{'all_GRAY_macro_geometry_chroma_nonworse':True}};exec(compile(ast.fix_missing_locations(ast.Module(body=[fs],type_ignores=[])),'<exact-final-screen>','exec'),scope);need(scope['final_screen']['prospective_numeric_screen_pass']==(cycle==32 and stop is None),'Final32 eligibility '+str(cycle)+'/'+str(stop))
 # Guard keeps strict scientific invariants for dispatch and permissive resource-only failure retention.
 class FakeCuda:
  @staticmethod
  def memory_allocated():return 0
 class Budget:
  def guard(self,**kwargs):pass
 class Net:
  bad=False
  def modules(self):return [types.SimpleNamespace(training=self.bad)]
  def named_parameters(self):return [('encoder.a',types.SimpleNamespace(requires_grad=self.bad,grad=torch.ones(1) if self.bad else None))]
 fake=Net();scope={'budget':Budget(),'counts':{'forward_slots':0,'backward_calls':0,'predictor_optimizer_steps':0,'empty_bypasses':0},'typed':{k:0 for k in contract.TYPED_CAPS},'TYPED_CAPS':contract.TYPED_CAPS,'torch':types.SimpleNamespace(cuda=FakeCuda()),'CONTRACT':contract.CONTRACT,'net':fake,'predictor_name':contract.predictor_name,'need':need};guard=extract(nested['guard'],scope);guard();fake.bad=True
 try:guard()
 except ValueError:need(True,'Normal guard rejects scientific ownership/mode failure')
 else:raise ValueError('Normal guard ignored failure')
 guard(terminal_save=True);need(True,'Terminal guard permits honest failed-state retention under resource limits')
 # Gray guard evaluates both scales, macro and every geometry; no recoloring.
 rows=[{'id':str(i),'style':'GRAY','mask_kind':contract.KINDS[i%4],'GRAY_raw_mean_channel_spread':.02,'GRAY_PNG_mean_channel_spread':5.} for i in range(8)];base={'records':rows};now={'records':[dict(r) for r in rows]};need(contract.compare_gray(now,base)['all_GRAY_macro_geometry_chroma_nonworse'],'GRAY exact baseline passes');now['records'][0]['GRAY_raw_mean_channel_spread']+=.01;now['records'][0]['GRAY_PNG_mean_channel_spread']+=1;result=contract.compare_gray(now,base);need(not result['all_GRAY_macro_geometry_chroma_nonworse'] and len(result['nonworse_failures'])==4,'GRAY raw/PNG macro+geometry regression fails')
 need(source.read_bytes()==initial_source_bytes,'Exact worker bytes unchanged throughout semantic audit')
 output=ROOT/'research/completion_codeformer_predictor_ce_source_audit_v1/independent_source_semantics.json';need(not output.exists(),'Preserve existing semantic receipt')
 files=[pin(f) for f in HERE.glob('*.py') if not f.name.startswith(('source_tests','cpu_clone'))]
 report={'format':'independent-predictor-CE-source-semantics-v1','complete':True,'independent':True,'UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(__file__),'source_files':files,'worker_AST_sha256':hashlib.sha256(ast.dump(worker,include_attributes=False).encode()).hexdigest(),'tested_nested_function_AST_sha256':{name:hashlib.sha256(ast.dump(nested[name],include_attributes=False).encode()).hexdigest() for name in ('cpu_copy','gradient_receipt','guard','save_state','forward','evaluate')},'architecture':pin(arch),'checks_passed':len(checks),'checks':checks,'CE_plain_tensor_witnesses':ce_checks,'scope':'Static graph and exact isolated helpers/plain CPU tensor semantics only; no checkpoint learning or actual VM preflight.','local_actions':{'model_objects':0,'NN_module_objects':0,'optimizer_objects':0,'optimizer_steps':0,'neural_model_calls':0,'plain_leaf_logits_gradient_queries':4,'actual_checkpoint_reads':0,'VM_calls':0,'training':False},'runtime_proof_limits':['No CodeFormer model instantiated, so full GPU gradient/update behavior and timing remain actual manual VM evidence.','Packet/source closure and actual transfer CLI guards require a separately staged protocol.','A qualifying numerical screen still requires all returned raw/PNG/state and planned visual sheets plus user/independent final review.'],'source_semantics_admitted':True,'packet_admitted':False,'ready_to_upload':False,'model_qualified':False,'seconds':time.monotonic()-start}
 output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'path':str(output),'sha256':pin(output)['sha256'],'checks':len(checks),'seconds':report['seconds']}))
if __name__=='__main__':main()

"""Independent source/dispatch/write-cap/failure-state fixtures; no neural runtime.
Extracts actual nested helpers and uses fake numerical-path callbacks/plain CPU tensors.
"""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import ast,copy,hashlib,importlib.util,json,math,os,random,sys,time,types,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'research/completion_codeformer_spatial_code_intervention_v1_development'
WORKER=HERE/'cctv_completion_codeformer_spatial_code_intervention_v1_vm.py'
CHECKS=0

def need(ok,msg):
 global CHECKS
 CHECKS+=1
 if not bool(ok):raise ValueError(msg)

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def pin(path):return {'path':str(Path(path).resolve()),'bytes':Path(path).stat().st_size,'sha256':sha(path)}
def load(name):
 package='research.completion_codeformer_spatial_code_intervention_v1';m=types.ModuleType(package);m.__path__=[str(HERE)];sys.modules.setdefault(package,m)
 spec=importlib.util.spec_from_file_location(package+'.'+name,HERE/(name+'.py'));module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module

def main():
 startpins={p.name:pin(p) for p in [WORKER,*[HERE/(k+'.py') for k in ('contracts','optimized_budget','probe_forward','app_delivery')]]};source=WORKER.read_text();tree=ast.parse(source,feature_version=(3,10));worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker');nested=lambda name:copy.deepcopy(next(n for n in ast.walk(worker) if isinstance(n,ast.FunctionDef) and n.name==name))
 for p in startpins.values():
  parsed=ast.parse(Path(p['path']).read_text(),feature_version=(3,10));need(not any(isinstance(n,ast.Attribute) and n.attr=='file_digest' for n in ast.walk(parsed)),'VM3.10 hash/syntax interface')
 need(not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('backward','Adam','SGD','grad') for n in ast.walk(worker)),'No optimizer/autograd calls in actual worker');need('torch.optim' not in source and 'requires_grad_(True)' not in source,'No hidden trainable path');need(len([n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.save'])==1 and any(isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.save' for n in ast.walk(nested('retain_failed_actual_state'))),'Only failure-capture Torch save');verify=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify');need(not any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(verify)),'Transfer verifier has no scientific imports')
 need("p['parity_cases']==design['data']['parity_cases_exact']" in source and "r['model_raw512_max_error']<=1e-6" in source,'Exact paritymembership/raw512gate corrections');need('old_input=app_input' not in source and 'old_input512_not_stored_reconstructed_only' not in source,'No invented historical512bridge evidence');need("'input512'} if bridge" not in source and 'spatial_effects(all_summaries)' in source,'Fixed9fieldbridge/all4additiveeffects')
 c=load('contracts');b=load('optimized_budget');bridge=load('probe_forward');need(c.TYPED_CAPS=={'parity':8,'bridge':4,'H':48,'V':48} and sum(c.TYPED_CAPS.values())==108,'All108typed caps');parent=Path(__file__).parent;fixtures=parent/'runtime_fixture_payloads';need(not fixtures.exists(),'Keep previous independent fixtures');fixtures.mkdir();paths=[]
 def budget(name):
  out=fixtures/name/'outputs';out.mkdir(parents=True);clock=[0.];bud=b.OutputBudget(out,0.,clock=lambda:clock[0],disk_free=lambda:100*1024**3);return bud,clock
 bud,clock=budget('dispatch');counts={'forward_slots':0,'actual_whole_model_forward_calls':0,'backward_calls':0,'gradient_queries':0,'optimizer_objects':0,'optimizer_steps':0,'teacher_neural_calls':0};typed={k:0 for k in c.TYPED_CAPS};witness={'quantizer':0,'generator_starts':0,'generator_completions':0};fakeparam=types.SimpleNamespace(requires_grad=False,grad=None);fakemodule=types.SimpleNamespace(training=False);grad=[False];cuda=[0];env={'budget':bud,'need':need,'CONTRACT':c.CONTRACT,'TYPED_CAPS':c.TYPED_CAPS,'counts':counts,'typed':typed,'witness':witness,'torch':types.SimpleNamespace(is_grad_enabled=lambda:grad[0],cuda=types.SimpleNamespace(memory_allocated=lambda:cuda[0])),'net':types.SimpleNamespace(parameters=lambda:[fakeparam],modules=lambda:[fakemodule])}
 for name in ('guard','call'):exec(compile(ast.Module(body=[nested(name)],type_ignores=[]),'<actual dispatch/guard on fake numerical paths>','exec'),env)
 callbacks=[]
 def fakework():
  callbacks.append(1);counts['actual_whole_model_forward_calls']+=1
  for k in witness:witness[k]+=1
  return 123
 for kind,n in c.TYPED_CAPS.items():
  for i in range(n):need(env['call'](kind,fakework)==123,'Actual boundeddispatch callback return unchanged')
 need(len(callbacks)==108 and all(v==108 for v in witness.values()),'All108fake numerical witnessesjoin dispatch')
 def refused(name,fn):
  before=len(callbacks)
  try:fn()
  except (ValueError,AssertionError):pass
  else:raise AssertionError('Expected beforedispatch refusal '+name)
  need(len(callbacks)==before,'Refused before any numerical callback '+name)
 refused('typedH49',lambda:env['call']('H',fakework));refused('unknownkind',lambda:env['call']('unknown',fakework));counts['forward_slots']=counts['actual_whole_model_forward_calls']=120;typed['H']=0
 refused('global121',lambda:env['call']('H',fakework));counts['forward_slots']=counts['actual_whole_model_forward_calls']=0
 grad[0]=True;refused('grad-enabled',lambda:env['call']('H',fakework));grad[0]=False;fakeparam.requires_grad=True;refused('trainableparameter',lambda:env['call']('H',fakework));fakeparam.requires_grad=False;fakeparam.grad=object();refused('unexpectedgradient',lambda:env['call']('H',fakework));fakeparam.grad=None;fakemodule.training=True;refused('trainingmode',lambda:env['call']('H',fakework));fakemodule.training=False;cuda[0]=21*1024**3;refused('CUDAcap',lambda:env['call']('H',fakework));cuda[0]=0
 limits=[]
 for name,relative,cap in [('HV','ORACLE_H_mask_touched_same_codebook/monitor/fixture.npz',6*1024**2),('parity','fresh_app_parity/fixture.npz',16*1024**2),('metadata','fixture.json',1024**2),('PNG','fixture.png',256*1024),('failure','failed_actual_model_state.pth.partial',384*1024**2)]:
  actual,now=budget('limit_'+name);path=actual.out/relative
  with actual.open(path,terminal=name=='failure',emergency=name=='failure') as f:
   f.write(b'KEEP');need(f.tell()==4,'Actual fourbyteprefix written')
   try:actual.before_entry_write(f.entry,cap+1,terminal=name=='failure',emergency=name=='failure')
   except ValueError:pass
   else:raise AssertionError('Actual bytecap absent '+name)
   need(f.tell()==4 and path.read_bytes()==b'KEEP','Prewrite limit retains existing bytes '+name)
  limits.append({'kind':name,'cap_bytes':cap,'retained_prefix_exact':True})
 actual,now=budget('deadline');now[0]=840
 try:actual.guard()
 except ValueError:pass
 else:raise AssertionError('Work deadline absent')
 actual.guard(emergency=True);now[0]=900
 try:actual.guard(emergency=True)
 except ValueError:pass
 else:raise AssertionError('Finish deadline absent')
 # Actual helper's plain additive effects: distinct per-case numbers prevent a constant-table false pass.
 summaries={};metrics=(*c.METRICS,'GRAY_raw_mean_channel_spread','GRAY_PNG_mean_channel_spread')
 for arm,offset in [('A',4.),('C',1.),('H',2.),('V',3.)]:
  for cohort in ('monitor','challenge'):
   summaries[(arm,cohort)]={'records':[{'id':cohort+str(i),'mask_kind':c.KINDS[i%4],'condition':'gray' if i<8 else 'rgb','style':'GRAY' if i<8 else 'RGB',**{key:offset+i/16+j/100 for j,key in enumerate(metrics)}} for i in range(32 if cohort=='monitor' else 16)]}
 effects=c.spatial_effects(summaries);effectchecks=0
 for cohort in ('monitor','challenge'):
  for row in effects['paired'][cohort]['cases']:
   for key,value in row['endpoints'].items():
    need(value['effects']=={'H_minus_A':value['values']['H']-value['values']['A'],'V_minus_A':value['values']['V']-value['values']['A'],'C_minus_A':value['values']['C']-value['values']['A'],'C_minus_H_minus_V_plus_A':value['values']['C']-value['values']['H']-value['values']['V']+value['values']['A']},'All48case4metric effects exact');effectchecks+=1
 need(len(effects['GRAY']['cases'])==8 and set(effects['GRAY']['geometry'])==set(c.KINDS),'All8GRAY2metric effects separately retained')
 # Execute actual source state/capture helper only on a fake497-scalar object.
 import torch
 torch.set_num_threads(1)
 def forbidden(*a,**k):raise AssertionError('NN/model/optimizer/autograd forbidden')
 torch.nn.Module.__init__=forbidden;torch.optim.Optimizer.__init__=forbidden;torch.autograd.grad=forbidden;torch.autograd.backward=forbidden
 values={'fixture_tensor_%03d'%i:torch.tensor([float(i)]) for i in range(497)};values['fixture_tensor_013'][0]=float('nan');fake=types.SimpleNamespace(state_dict=lambda:values,parameters=lambda:[],modules=lambda:[]);failurebudget,failclock=budget('actual_failure_toy');failclock[0]=850 # Scientific work deadline expired, recovery reserve remains.
 proxy=types.SimpleNamespace(save=torch.save,isfinite=torch.isfinite,get_rng_state=torch.get_rng_state,cuda=types.SimpleNamespace(get_rng_state_all=lambda:[torch.zeros(16,dtype=torch.uint8)]));fenv={'net':fake,'budget':failurebudget,'out':failurebudget.out,'torch':proxy,'hashlib':hashlib,'np':np,'random':random,'os':os,'zipfile':zipfile,'PurePosixPath':PurePosixPath,'need':need,'sha':sha,'CONTRACT':c.CONTRACT,'PARENT_LOGICAL':c.PARENT_LOGICAL,'ORIGINAL_SHA':c.ORIGINAL_SHA,'PARENT_SHA':c.PARENT_SHA,'expected':'0'*64,'counts':{'forward_slots':1},'typed':{'H':1},'progress':{'phase':'independent_toy_fixture_no_neural_work'},'witness':{'quantizer':1,'generator_starts':1,'generator_completions':0},'record':lambda p,v,emergency=False:failurebudget.json(p,v,emergency=emergency)}
 exec(compile(ast.Module(body=[nested('state')],type_ignores=[]),'<actual hash helper on497fake CPU tensors>','exec'),fenv);actual=fenv['state']();need(actual['state_tensors']==497 and not actual['all_tensors_finite'] and not actual['tensors']['fixture_tensor_013']['finite'],'Actualstate hashretainsNaNflagandbytes')
 fn=nested('retain_failed_actual_state');factory=ast.FunctionDef(name='factory',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[ast.Assign(targets=[ast.Name(id='failure_state_attempted',ctx=ast.Store())],value=ast.Constant(False)),fn,ast.Return(value=ast.Name(id='retain_failed_actual_state',ctx=ast.Load()))],decorator_list=[]);ast.fix_missing_locations(factory);exec(compile(ast.Module(body=[factory],type_ignores=[]),'<actual once-only failure capture with native nonlocal>','exec'),fenv);capture=fenv['factory']();capture(actual)
 receipt=json.loads((failurebudget.out/'failure_state_receipt.json').read_text());need(receipt['complete_committed_state'] and not receipt['partial_retained_not_usable'] and receipt['capture_error'] is None,'Actual boundedfailure save succeeds in60sresource-only reserve');path=failurebudget.out/'failed_actual_model_state.pth'
 with zipfile.ZipFile(path) as z:need(z.testzip() is None and len(z.infolist())<1024,'Actual497toyZIPcompleteCRC')
 saved=torch.load(path,map_location='cpu',weights_only=True);need(len(saved['state_dict'])==497 and torch.isnan(saved['state_dict']['fixture_tensor_013']).all().item() and saved['actual_state_hash_receipt']==actual and saved['actual_path_witness']==fenv['witness'] and saved['optimizer_updates']==0,'SafeCPUload preservesactualNaN/witness497toy state')
 try:capture(actual)
 except ValueError:pass
 else:raise AssertionError('Second failure capture was permitted')
 for p in startpins.values():need(pin(p['path'])==p,'No source drift during independent audit')
 result={'format':'independent-spatial-code-intervention-v1-runtime-source-semantics','complete':True,'independent':True,'UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(__file__),'source_pins':startpins,'checks_passed':CHECKS,'source_no_training_no_unbound_markers':True,'exactparitymembership_and_raw512gate':True,'historicalC_input512_truthful_no_reconstructedclaim':True,'actual_dispatch_helper':{'valid_fake_callbacks':108,'invalid_typed_global_gradient_mode_CUDA_cases_refuse_beforecallback':8,'allquantizer_generatorwitnesses_checked':True,'no_real_neural_function_called':True},'bounded_writer_prewrite_fixtures':limits,'cooperative840_hard900_reserve_enforced':True,'additive_effects':{'all48cases4metrics':effectchecks,'all8gray2metricseparate':True,'interactionformula_exact':True},'actual_failure_helper_toy_fixture':{'tensors':497,'NaN_actualbytes_flag_retained':True,'workingdeadline_expired850_resourcefinishstillpermitscapture':True,'actualsource_singleattempt_save_CRC_exclusivecommit':True,'CPU_weights_only_toy_loads':1,'secondattempt_refused':True,'payloads':[pin(p) for p in failurebudget.out.iterdir() if p.is_file()],'realcheckpoint_or_model':False},'prepared_runtime_admitted_as_source_only':True,'packet_transfer_review_still_required':True,'local_actions':{'real_model_constructions':0,'neural_forwards':0,'autograd_calls':0,'optimizer_objects_or_steps':0,'realcheckpoint_loads':0,'VM_calls':0,'returned_code_executed':False,'plainCPUinteger_and_fake497serialization_only':True},'model_qualified':False,'goal_complete':False}
 target=Path(__file__).parent/'independent_runtime_source_semantics.json'
 with target.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'complete':True,'receipt':pin(target),'checks':CHECKS,'neural_calls':0},indent=2),flush=True)
if __name__=='__main__':main()

"""Independent extracted zero-update failure-state and predispatch cap fixtures.
Only tiny plain CPU tensors/callbacks; no neural model/full forward/optimizer.
"""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,types
ROOT=Path(__file__).resolve().parents[2];WORKER=ROOT/'research/completion_codeformer_same_codebook_probe_v1_development/cctv_completion_codeformer_same_codebook_probe_v1_vm.py';OUT=ROOT/'research/completion_codeformer_same_codebook_probe_v1_audit/independent_failure_counter_fixtures.json'
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pin(p):return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)}
def main():
 need(sha(WORKER)=='7036fe80d6b942a22f7eb5bf2c662ec49a9e5e99d4b347ccb2a7b9eaa374b2f6','Worker changed');tree=ast.parse(WORKER.read_text());w=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker');nodes={n.name:n for n in w.body if isinstance(n,ast.FunctionDef)};checks=[]
 def check(ok,msg):need(ok,msg);checks.append(msg)
 import torch
 torch.set_num_threads(1);values={f'tensor{i:03d}':torch.tensor([float(i)],dtype=torch.float32) for i in range(497)}
 class TinyHolder:
  def state_dict(self):return values
  def parameters(self):return values.values()
  def modules(self):return [types.SimpleNamespace(training=False)]
 holder=TinyHolder();env={'net':holder,'torch':torch,'hashlib':hashlib,'need':need};exec(compile(ast.Module(body=[nodes['state']],type_ignores=[]),'<exact state helper>','exec'),env);state=env['state'];before=state();check(len(before['tensors'])==497 and before['all_tensors_finite'] and before['all_parameters_frozen'] and before['all_parameter_gradients_none'] and before['all_modules_eval'],'All497 finite tinyCPU state witness')
 values['tensor240'][0]=float('nan');failed=state();check(len(failed['tensors'])==497 and failed['all_tensors_finite'] is False and failed['tensors']['tensor240']['finite'] is False and all(failed['tensors'][n]['finite'] for n in values if n!='tensor240'),'Nonfinite failed state still retains all497 hashes/flags');check(failed['logical_sha256']!=before['logical_sha256'] and failed['tensors']['tensor240']['sha256']!=before['tensors']['tensor240']['sha256'],'Actual failed bytes affect state fingerprints');check(all(failed['tensors'][n]==before['tensors'][n] for n in values if n!='tensor240'),'All496 unaffected tensor records remain exact')
 counts={'forward_slots':220,'actual_whole_model_forward_calls':220};typed={'A':0};events=[]
 def guard():need(counts['forward_slots']<=220 and counts['forward_slots']==counts['actual_whole_model_forward_calls'],'Counter guard')
 env={'counts':counts,'typed':typed,'CONTRACT':{'forward_slots_max':220},'guard':guard,'need':need};exec(compile(ast.Module(body=[nodes['call']],type_ignores=[]),'<exact call helper>','exec'),env)
 def callback():events.append('called');counts['actual_whole_model_forward_calls']+=1;return 'exact_result'
 try:env['call']('A',callback)
 except ValueError:check(not events and counts['forward_slots']==220 and typed['A']==0,'At220 hardcap refuses before callback/increments')
 else:raise ValueError('221st call allowed')
 counts.update(forward_slots=219,actual_whole_model_forward_calls=219);value=env['call']('A',callback);check(value=='exact_result' and counts=={'forward_slots':220,'actual_whole_model_forward_calls':220} and typed['A']==1 and events==['called'],'At219 exactly one callback reaches220')
 check(sha(WORKER)=='7036fe80d6b942a22f7eb5bf2c662ec49a9e5e99d4b347ccb2a7b9eaa374b2f6','Worker stable through fixtures')
 report={'format':'independent-same-codebook-failure-state-and-counter-fixtures-v1','complete':True,'independent':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(Path(__file__)),'worker':pin(WORKER),'checks':checks,'checks_passed':len(checks),'actual_extracted_state_helper_AST_sha256':hashlib.sha256(ast.dump(nodes['state']).encode()).hexdigest(),'actual_extracted_call_helper_AST_sha256':hashlib.sha256(ast.dump(nodes['call']).encode()).hexdigest(),'local_actions':{'plain_tinyCPU_tensor_count':497,'full_neural_forward':0,'model_or_optimizer_objects':0,'gradients':0,'VM_calls':0},'runtime_or_packet_admitted':False,'model_qualified':False}
 need(not OUT.exists(),'Preserve prior fixtures');OUT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({'complete':True,'receipt':pin(OUT),'checks':len(checks)},indent=2))
if __name__=='__main__':main()

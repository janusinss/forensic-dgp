"""Independent owned-forward semantic identity and plain-index selection fixtures.
No neural model, complete forward, gradient, optimizer or VM execution.
"""
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,hashlib,importlib.util,json,sys
ROOT=Path(__file__).resolve().parents[2];DEV=ROOT/'research/completion_codeformer_same_codebook_probe_v1_development';OUT=ROOT/'research/completion_codeformer_same_codebook_probe_v1_audit/independent_bridge_semantics.json'
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pin(p):return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)}
def main():
 source=ROOT/'third_party/codeformer/codeformer_arch.py';owned=DEV/'probe_forward.py';checks=[]
 def check(ok,msg):need(ok,msg);checks.append(msg)
 check(sha(source)=='ee9758522173b72fc1f292df825b9a32179aebfd96142ececf9aa68f6e12e9eb','Exact admitted local CodeFormer architecture')
 original=ast.parse(source.read_text(),feature_version=(3,10));klass=next(n for n in original.body if isinstance(n,ast.ClassDef) and n.name=='CodeFormer');f=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='forward');tree=ast.parse(owned.read_text(),feature_version=(3,10));g=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='probe_forward')
 check(not any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(tree)),'Owned bridge imports no scientific/model modules')
 check([a.arg for a in g.args.kwonlyargs]==['F','torch','adaptive_instance_normalization','oracle_indices'],'All late scientific dependencies explicitly injected')
 norm=copy.deepcopy(g);norm.name='forward';norm.args.kwonlyargs=[];norm.args.kw_defaults=[];body=[];removed=[]
 for node in norm.body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='predicted_indices' for t in node.targets):removed.append('predicted_clone');continue
  if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='select_code_indices':removed.append('selected_override');continue
  if isinstance(node,ast.Return) and isinstance(node.value,ast.Tuple) and len(node.value.elts)==5:node.value.elts=node.value.elts[:3];removed.append('extended_return')
  body.append(node)
 norm.body=body;check(removed==['predicted_clone','selected_override','extended_return'],'Exactly three diagnostic-only AST insertions')
 check(ast.dump(norm,include_attributes=False)==ast.dump(f,include_attributes=False),'Full ordinary forward AST identical after removing override/observations')
 statements=g.body;idx=next(i for i,n in enumerate(statements) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='select_code_indices');check(isinstance(statements[idx+1],ast.Assign) and 'get_codebook_feat' in ast.unparse(statements[idx+1]),'Override occurs immediately before same-codebook vector lookup')
 check('enc_feat_dict[f_size].detach()' in ast.unparse(g) and 'quant_feat.detach()' in ast.unparse(g),'Observed encoder feature and quantized detach semantics intact')
 check(not any(isinstance(n,ast.Attribute) and n.attr in ('backward','step','load_state_dict') for n in ast.walk(tree)),'Bridge has no training/state replacement operations')
 spec=importlib.util.spec_from_file_location('independent_same_codebook_bridge',owned);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);check(not any(n=='torch' or n.startswith('torch.') for n in sys.modules),'Pure bridge module import does not load Torch')
 import torch
 torch.set_num_threads(1);predicted=torch.tensor([[[0],[2],[511]]],dtype=torch.int64);oracle=torch.tensor([[[511],[7],[0]]],dtype=torch.int64);before=predicted.clone();oracle_before=oracle.clone();actual=m.select_code_indices(predicted,None,codebook_size=512)
 check(actual is predicted and torch.equal(actual,before),'No oracle returns identical predicted indices without copy/change')
 used=m.select_code_indices(predicted,oracle,codebook_size=512);check(torch.equal(used,oracle) and used.data_ptr()!=oracle.data_ptr() and used.data_ptr()!=predicted.data_ptr() and used.grad_fn is None,'Oracle is an exact detached independent clone')
 check(torch.equal(predicted,before) and torch.equal(oracle,oracle_before),'Selection never mutates actual prediction or label');used[0,0,0]=10;check(torch.equal(oracle,oracle_before) and torch.equal(predicted,before),'Returned oracle mutation cannot affect source/prediction')
 cases=[('shape',oracle.squeeze(-1)),('dtype',oracle.to(torch.int32)),('negative',torch.tensor([[[-1],[0],[1]]],dtype=torch.int64)),('too_large',torch.tensor([[[512],[0],[1]]],dtype=torch.int64))]
 for name,value in cases:
  try:m.select_code_indices(predicted,value,codebook_size=512)
  except ValueError:check(True,'Invalid '+name+' override refused')
  else:raise ValueError('Invalid '+name+' accepted')
 check(sha(owned)==pin(owned)['sha256'],'Owned source stable through tests')
 report={'format':'independent-same-codebook-owned-bridge-semantic-admission-v1','complete':True,'independent':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(Path(__file__)),'original_architecture':pin(source),'owned_bridge':pin(owned),'checks':checks,'checks_passed':len(checks),'ordinary_path_entire_forward_AST_exact':True,'only_selected_code_override_plus_predicted_used_observation':True,'plain_CPU_index_selection_tests_only':True,'full_probe_forward_not_executed':True,'actual_vm_parity_still_required':{'fresh_pairs':4,'initial48A_raw_tolerance':1e-6,'initial48A_PNG_byte_exact':True},'local_actions':{'Torch_imported_for_plain_tensor_index_fixtures':True,'neural_modules_or_forwards':0,'gradients':0,'optimizer_objects_or_steps':0,'VM_calls':0},'runtime_or_packet_admitted':False,'model_qualified':False}
 need(not OUT.exists(),'Preserve prior bridge admission');OUT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({'complete':True,'receipt':pin(OUT),'checks':len(checks)},indent=2))
if __name__=='__main__':main()

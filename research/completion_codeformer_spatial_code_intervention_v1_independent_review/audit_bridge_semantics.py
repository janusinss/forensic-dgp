"""Independent owned-forward AST / CPU integer selector and geometric-grid fixtures.
Plain integer tensors only: no neural module/model, gradients, weights or VM.
"""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,importlib.util,io,json,os,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'research/completion_codeformer_spatial_code_intervention_v1_development/probe_forward.py'
STAGE=ROOT/'outputs/completion_codeformer_same_codebook_probe_v1_packet/draft_001/cctv_completion_codeformer_same_codebook_probe_v1_vm'
ORIGINAL=STAGE/'third_party/codeformer/codeformer_arch.py'
OLD=ROOT/'outputs/completion_codeformer_same_codebook_probe_v1_return/cctv_completion_codeformer_same_codebook_probe_v1_results'
CHECKS=0

def need(ok,msg):
 global CHECKS
 CHECKS+=1
 if not bool(ok):raise ValueError(msg)

def ip(p):
 p=Path(p).resolve();return Path('\\\\?\\'+str(p)) if os.name=='nt' and not str(p).startswith('\\\\?\\') else p

def sha(p):
 h=hashlib.sha256()
 with ip(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def pin(p):return {'path':str(Path(p).resolve()),'bytes':ip(p).stat().st_size,'sha256':sha(p)}
def read(p):return json.loads(ip(p).read_text(encoding='utf-8-sig'))
def loadcode(p,key):
 with zipfile.ZipFile(ip(p)) as z:v=np.load(io.BytesIO(z.read(key+'.npy')),allow_pickle=False,max_header_size=512)
 need(v.dtype==np.int64 and v.shape==(1,256),'Exact retained code array');return v

def main():
 need(sha(ORIGINAL)=='ee9758522173b72fc1f292df825b9a32179aebfd96142ececf9aa68f6e12e9eb','Pinnedoriginal CodeFormer architecture');official=ast.parse(ip(ORIGINAL).read_text());owned=ast.parse(ip(SOURCE).read_text());old=next(n for c in official.body if isinstance(c,ast.ClassDef) and c.name=='CodeFormer' for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='forward');new=next(n for n in owned.body if isinstance(n,ast.FunctionDef) and n.name=='probe_forward')
 need([a.arg for a in new.args.kwonlyargs]==['F','torch','adaptive_instance_normalization','oracle_indices','grid_mask'],'Only explicit owned dependencies/oracle-grid signature extras')
 removed=[];body=[]
 for n in new.body:
  if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('predicted_indices','top_idx'):
   if n.targets[0].id=='predicted_indices':need(ast.unparse(n.value)=='top_idx.clone()','Only predicted index witness copy');removed.append('prediction_clone');continue
   if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='select_code_indices':
    need([ast.unparse(a) for a in n.value.args]==['top_idx','oracle_indices'] and {k.arg:ast.unparse(k.value) for k in n.value.keywords}=={'codebook_size':'512','grid_mask':'grid_mask','torch':'torch'},'Only samebookselector substitution');removed.append('samebook_grid_selector');continue
  if isinstance(n,ast.Return) and isinstance(n.value,ast.Tuple) and len(n.value.elts)==5:
   need([ast.unparse(x) for x in n.value.elts]==['out','logits','lq_feat','predicted_indices','top_idx.clone()'],'Only prediction/used witness returnextras');n.value.elts=n.value.elts[:3];removed.append('return_witnesses')
  body.append(n)
 need(removed==['prediction_clone','samebook_grid_selector','return_witnesses'],'Exactly3 authorized forward interventions');need(ast.dump(ast.Module(body=body,type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=old.body,type_ignores=[]),include_attributes=False),'Owned ordinary entire computation AST equals original afterexact3 authorized additions')
 # Import only the owned pure module; model architecture is never imported.
 spec=importlib.util.spec_from_file_location('independent_owned_spatial_bridge',SOURCE);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 import torch
 torch.set_num_threads(1)
 # Prevent accidentally constructing a neural module or querying autograd in this fixture process.
 def forbidden(*a,**k):raise AssertionError('Neural/autograd/optimizer work forbidden in integer selector audit')
 torch.nn.Module.__init__=forbidden;torch.autograd.grad=forbidden;torch.autograd.backward=forbidden;torch.optim.Optimizer.__init__=forbidden
 from PIL import Image
 p=read(OLD/'protocol.json');labels={r['teacher_key']:r for r in p['oracle_labels']};actual=[]
 with torch.no_grad():
  for cohort,key in [('monitor','monitor_cases'),('challenge','challenge_cases')]:
   for case in p[key]:
    with Image.open(ip(STAGE/case['mask'])) as im:m=(np.asarray(im)==255)[None,None].astype(np.float32)
    grid=module.geometric_touched_grid(m,np=np);manual=np.array([bool(m[0,0,y:y+16,x:x+16].any()) for y in range(0,256,16) for x in range(0,256,16)])[None,:];need(grid.dtype==np.bool_ and grid.shape==(1,256) and np.array_equal(grid,manual),'All48 actualmask row-major geometry independentlysliced');pred=loadcode(OLD/'outputs/ACTUAL_A_R1_w1'/cohort/(case['id']+'.npz'),'predicted_codes');oracle=loadcode(STAGE/labels[case['teacher_key']]['path'],'nearest_codes');pt=torch.from_numpy(pred.copy()).unsqueeze(-1);ot=torch.from_numpy(oracle.copy()).unsqueeze(-1);gt=torch.from_numpy(grid.copy()).unsqueeze(-1)
    for name,sel in [('H',gt),('V',~gt)]:
     result=module.select_code_indices(pt,ot,codebook_size=512,grid_mask=sel,torch=torch);expected=np.where(grid if name=='H' else ~grid,oracle,pred);need(np.array_equal(result.squeeze(-1).numpy(),expected) and torch.equal(pt,torch.from_numpy(pred).unsqueeze(-1)) and torch.equal(ot,torch.from_numpy(oracle).unsqueeze(-1)),'All48 actualselectedH/V codes sources correct/inputsunmodified');need(result.dtype==torch.int64 and result.shape==(1,256,1) and not result.requires_grad and result.data_ptr() not in (pt.data_ptr(),ot.data_ptr()),'Independent detached integer output noalias')
    actual.append({'id':case['id'],'cohort':cohort,'touched':int(grid.sum()),'visible':256-int(grid.sum()),'H_V_sources_exact':True})
  for label,m in [('empty',np.zeros((1,1,256,256),np.float32)),('full',np.ones((1,1,256,256),np.float32)),('lastpixel',np.pad(np.ones((1,1,1,1),np.float32),((0,0),(0,0),(255,0),(255,0)))),('firstpixel',np.pad(np.ones((1,1,1,1),np.float32),((0,0),(0,0),(0,255),(0,255))))]:
   g=module.geometric_touched_grid(m,np=np);want=np.zeros((1,256),bool)
   if label=='full':want[:]=True
   elif label=='lastpixel':want[0,255]=True
   elif label=='firstpixel':want[0,0]=True
   need(np.array_equal(g,want),'Gridedge fixture '+label)
  need(module.select_code_indices(pt,None,codebook_size=512,torch=torch) is pt,'Nooracleordinary path exactobject');allcodes=module.select_code_indices(pt,ot,codebook_size=512,torch=torch);need(torch.equal(allcodes,ot) and allcodes.data_ptr()!=ot.data_ptr(),'FullC codecopy exact/noalias')
  rejections=[]
  for name,args in [
   ('oracledtype',dict(oracle_indices=ot.float(),grid_mask=gt)),('oracleshape',dict(oracle_indices=ot.squeeze(-1),grid_mask=gt)),('oracledevice',dict(oracle_indices=torch.empty(ot.shape,dtype=torch.int64,device='meta'),grid_mask=gt)),('negativeoracle',dict(oracle_indices=torch.full_like(ot,-1),grid_mask=gt)),('oracle512',dict(oracle_indices=torch.full_like(ot,512),grid_mask=gt)),('griddtype',dict(oracle_indices=ot,grid_mask=gt.to(torch.uint8))),('gridshape',dict(oracle_indices=ot,grid_mask=gt.squeeze(-1))),('gridwithoutoracle',dict(oracle_indices=None,grid_mask=gt))]:
   try:module.select_code_indices(pt,codebook_size=512,torch=torch,**args)
   except ValueError:rejections.append(name)
   else:raise AssertionError('Invalid selector inputaccepted '+name)
  need(len(rejections)==8,'All8invalid selector contracts rejected')
  for bad in [np.full((1,1,256,256),np.nan,np.float32),np.full((1,1,256,256),.5,np.float32),np.zeros((256,256),np.float32)]:
   try:module.geometric_touched_grid(bad,np=np)
   except ValueError:pass
   else:raise AssertionError('Invalid geometric masks accepted')
 report={'format':'independent-spatial-code-intervention-v1-pure-bridge-semantics','complete':True,'independent':True,'UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(__file__),'owned_helper':pin(SOURCE),'pinned_original_architecture':pin(ORIGINAL),'checks_passed':CHECKS,'ordinary_forward_normalized_AST_exact_after_only3authorizedwitness_selector_additions':True,'all48actual_reviewed_masks_and96_HVintegerselections_exact':True,'percase':actual,'edge_gridfixtures':4,'invalid_selector_rejections':rejections,'fullC_noneordinary_copy_alias_contracts_pass':True,'invalid_geometric_masks_rejected':3,'CPU_tensor_runtime':str(torch.__version__),'scope':'Pure integerwhere/grid and AST fixtures only. Does not run or reproduce a neural model; freshactual parity/Cbridge stillrequired onVM. Geometric cells are notreceptivefields oridentityaccuracy.','local_actions':{'neural_modules_or_models':0,'neural_forward_calls':0,'autograd_calls':0,'optimizer_objects_or_steps':0,'checkpoint_loads':0,'VM_calls':0,'actual_returned_code_executed':False},'packet_or_transfer_ready':False,'model_qualified':False}
 target=Path(__file__).parent/'independent_bridge_semantics.json'
 with target.open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'complete':True,'receipt':pin(target),'checks':CHECKS,'actual_case_integer_fixtures':48,'neural_calls':0},indent=2),flush=True)
if __name__=='__main__':main()

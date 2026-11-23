"""Source/serialization/plain-value tests. No model, NN forward or real checkpoint."""
from pathlib import Path
import ast,contextlib,copy,hashlib,importlib.util,io,json,runpy,sys,tempfile,time,types,unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;TMP=Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs')
PACKAGE='research.completion_codeformer_spatial_code_intervention_v1';package=types.ModuleType(PACKAGE);package.__path__=[str(HERE)];sys.modules[PACKAGE]=package

def load(name):
 spec=importlib.util.spec_from_file_location(PACKAGE+'.'+name,HERE/(name+'.py'));m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
c=load('contracts');b=load('optimized_budget');bridge=load('probe_forward');delivery=load('app_delivery')
WORKER=HERE/'cctv_completion_codeformer_spatial_code_intervention_v1_vm.py';SOURCE=WORKER.read_text();TREE=ast.parse(SOURCE);WN=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
def original_forward():
 tree=ast.parse((ROOT/'third_party/codeformer/codeformer_arch.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CodeFormer');return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
def nested(name):return next(n for n in ast.walk(WN) if isinstance(n,ast.FunctionDef) and n.name==name)
class SourceTests(unittest.TestCase):
 def test_01_runtime_Python310_no_newer_hash_helper(self):
  for name in ('contracts.py','optimized_budget.py','probe_forward.py','app_delivery.py',WORKER.name):
   tree=ast.parse((HERE/name).read_text(),feature_version=(3,10));self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr=='file_digest' for n in ast.walk(tree)))
  self.assertNotIn('REPLACEMENT',SOURCE)
 def test_02_actual_help_blocks_scientific_imports(self):
  original=__import__
  def blocked(name,*args,**kwargs):
   if name.split('.')[0] in ('torch','numpy','PIL'):raise AssertionError('Scientific import in help '+name)
   return original(name,*args,**kwargs)
  with patch('builtins.__import__',blocked),patch.object(sys,'argv',[str(WORKER),'--help']),contextlib.redirect_stdout(io.StringIO()):
   with self.assertRaises(SystemExit) as result:runpy.run_path(str(WORKER),run_name='__main__')
  self.assertEqual(result.exception.code,0)
 def test_03_zero_training_no_constructor_or_backwards_in_transfer(self):
  self.assertNotIn('torch.optim',SOURCE);self.assertNotIn('requires_grad_(True)',SOURCE)
  for node in ast.walk(WN):
   if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):self.assertNotIn(node.func.attr,('backward','Adam','SGD','grad'))
  saves=[n for n in ast.walk(TREE) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='save' and isinstance(n.func.value,ast.Name) and n.func.value.id=='torch'];self.assertEqual(len(saves),1);self.assertIn(saves[0],list(ast.walk(nested('retain_failed_actual_state'))))
  verify=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='verify');self.assertFalse(any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(verify)))
 def test_04_exact_resources_and_declared_counts(self):
  self.assertEqual(c.TYPED_CAPS,{'parity':8,'bridge':4,'H':48,'V':48});self.assertEqual(c.CONTRACT['forward_slots_planned'],sum(c.TYPED_CAPS.values()));self.assertEqual(c.CONTRACT['forward_slots_max'],120);self.assertEqual(c.CONTRACT['new_NPZ_max'],104);self.assertEqual(c.CONTRACT['new_PNG_max'],96)
  self.assertEqual((c.CONTRACT['worker_seconds'],c.CONTRACT['cooperative_stop_seconds']),(900,840));self.assertEqual(c.CONTRACT['minimum_free_bytes'],6*1024**3);self.assertEqual(c.CONTRACT['terminal_bytes'],384*1024**2);self.assertEqual(c.CONTRACT['output_bytes'],1024**3)
 def test_05_closed_unique_output_namespace(self):
  p={'parity_cases':[{'id':'p'+str(i)} for i in range(4)],'bridge_cases':[{'id':'b'+str(i)} for i in range(4)],'monitor_cases':[{'id':'m'+str(i)} for i in range(32)],'challenge_cases':[{'id':'c'+str(i)} for i in range(16)]};paths=c.output_paths(p)
  self.assertEqual(len(paths),len(set(paths)));self.assertLess(len(paths)+4,400);self.assertEqual(sum(v.endswith('.npz') for v in paths),104);self.assertEqual(sum(v.endswith('.png') for v in paths),96);self.assertIn('outputs/failed_actual_model_state.pth.partial',paths);self.assertNotIn('outputs/terminal_training_state.pth',paths)
 def test_06_exact_single_cell_geometric_any_rule(self):
  m=np.zeros((1,1,256,256),np.float32);m[0,0,0,0]=1;m[0,0,15,15]=1;m[0,0,16,31]=1;m[0,0,255,255]=1;grid=bridge.geometric_touched_grid(m,np=np)
  self.assertEqual(np.flatnonzero(grid).tolist(),[0,17,255]);self.assertEqual(grid.dtype,np.bool_)
  for bad in (m[:,:,1:],m+0.1,np.full(m.shape,np.nan)):
   with self.assertRaises(ValueError):bridge.geometric_touched_grid(bad,np=np)
 def test_07_original_forward_AST_only_index_override(self):
  fn=copy.deepcopy(next(n for n in ast.parse((HERE/'probe_forward.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='probe_forward'));fn.name='forward';fn.args.kwonlyargs=[];fn.args.kw_defaults=[]
  fn.body=[node for node in fn.body if not (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('predicted_indices','top_idx') for t in node.targets) and (any(isinstance(t,ast.Name) and t.id=='predicted_indices' for t in node.targets) or isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='select_code_indices'))];fn.body[-1].value.elts=fn.body[-1].value.elts[:3]
  self.assertEqual(ast.dump(fn),ast.dump(original_forward()))
 def test_08_H_V_integer_exclusivity_and_alias_safety(self):
  import torch
  predicted=torch.arange(256,dtype=torch.int64).reshape(1,256,1);oracle=511-predicted;grid=predicted%3==0
  h=bridge.select_code_indices(predicted,oracle,codebook_size=512,grid_mask=grid,torch=torch);v=bridge.select_code_indices(predicted,oracle,codebook_size=512,grid_mask=~grid,torch=torch)
  self.assertTrue(torch.equal(h[grid],oracle[grid]));self.assertTrue(torch.equal(h[~grid],predicted[~grid]));self.assertTrue(torch.equal(v[~grid],oracle[~grid]));self.assertTrue(torch.equal(v[grid],predicted[grid]));oracle.fill_(0);self.assertTrue(h.max()>0)
  for value in (predicted.float(),predicted.squeeze(-1),predicted+512,predicted-512):
   with self.assertRaises(ValueError):bridge.select_code_indices(predicted,value,codebook_size=512,grid_mask=grid,torch=torch)
  with self.assertRaises(ValueError):bridge.select_code_indices(predicted,None,codebook_size=512,grid_mask=grid,torch=torch)
  with self.assertRaises(ValueError):bridge.select_code_indices(predicted,predicted,codebook_size=512,grid_mask=grid.float(),torch=torch)
 def test_09_actual_quantizer_witness_is_passthrough(self):
  observed=[];sentinel=object();witness={'quantizer':0,'generator_starts':0,'generator_completions':0}
  def original(*args,**kwargs):observed.append((args,kwargs));return sentinel
  env={'witness':witness,'original_quantize':original}
  for name in ('counted_quantize','generator_start','generator_done'):exec(compile(ast.Module(body=[nested(name)],type_ignores=[]),'<actual witness plaincallbacks>','exec'),env)
  self.assertIs(env['counted_quantize']('indices',shape=[1,16,16,256]),sentinel);self.assertEqual(observed,[(('indices',),{'shape':[1,16,16,256]})]);env['generator_start'](None,None);env['generator_done'](None,None,None);self.assertEqual(witness,{'quantizer':1,'generator_starts':1,'generator_completions':1})
  self.assertLess(SOURCE.index('net.quantize.get_codebook_feat=counted_quantize'),SOURCE.index("progress['phase']='four_fresh_owned_vs_original_app_parity'"));self.assertIn("delattr(net.quantize,'get_codebook_feat')",SOURCE)
 def test_10_typed_and_global_caps_refuse_before_neural(self):
  fn=nested('call');counts={'forward_slots':0};typed={k:0 for k in c.TYPED_CAPS};witness={'quantizer':0,'generator_starts':0,'generator_completions':0};env={'guard':lambda:None,'need':c.need,'TYPED_CAPS':c.TYPED_CAPS,'CONTRACT':c.CONTRACT,'counts':counts,'typed':typed,'witness':witness,'torch':types.SimpleNamespace(is_grad_enabled=lambda:False),'net':types.SimpleNamespace(parameters=lambda:[],modules=lambda:[])};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual dispatch helper>','exec'),env);called=[]
  typed['parity']=8
  with self.assertRaises(ValueError):env['call']('parity',lambda:called.append(True))
  self.assertEqual(counts['forward_slots'],0);self.assertEqual(called,[])
  typed['parity']=0;counts['forward_slots']=120
  with self.assertRaises(ValueError):env['call']('parity',lambda:called.append(True))
  self.assertEqual(counts['forward_slots'],120);self.assertEqual(typed['parity'],0)
 def budget(self,path,clock=lambda:0):
  out=Path(path)/'outputs';out.mkdir();return b.OutputBudget(out,0,clock=clock,disk_free=lambda:100*1024**3)
 def test_11_native_numeric_PNG_bounded_callbacks(self):
  with tempfile.TemporaryDirectory(dir=TMP) as t:
   budget=self.budget(t);p=budget.out/'example.npz'
   with budget.open(p) as stream:np.savez_compressed(stream,result=np.zeros((1,3,256,256),np.float32))
   with np.load(p,allow_pickle=False) as z:self.assertEqual(z['result'].shape,(1,3,256,256))
   with budget.open(budget.out/'image.png') as stream:Image.fromarray(np.zeros((256,256,3),np.uint8)).save(stream,format='PNG')
   budget.assert_ledger_consistent();self.assertEqual(budget.total(),sum(p.stat().st_size for p in budget.out.iterdir()))
 def test_12_limits_refuse_before_writes_preserve_prefix(self):
  with tempfile.TemporaryDirectory(dir=TMP) as t:
   now=[0];budget=self.budget(t,lambda:now[0]);p=budget.out/'partial.npz'
   with budget.open(p) as stream:
    stream.write(b'keep');self.assertEqual(stream.tell(),4)
    with self.assertRaises(ValueError):stream.write(b'x'*(6*1024**2))
    self.assertEqual(stream.tell(),4);now[0]=840
    with self.assertRaises(ValueError):stream.write(b'bad')
   self.assertEqual(p.read_bytes(),b'keep');budget.guard(emergency=True);now[0]=900
   with self.assertRaises(ValueError):budget.guard(emergency=True)
 def test_13_source_paths_metadata_cap_checkpoint_scope(self):
  with tempfile.TemporaryDirectory(dir=TMP) as t:
   budget=self.budget(t)
   for name in ('other.pth','any.pt.partial','../escape.npz'):
    with self.assertRaises(ValueError):budget.open(budget.out/name)
   path=budget.out/'bounded.json'
   with budget.open(path) as stream:
    with self.assertRaises(ValueError):stream.write(b'x'*(1024**2+1))
   self.assertEqual(path.stat().st_size,0)
   with budget.open(budget.out/'failed_actual_model_state.pth.partial',terminal=True,emergency=True) as stream:stream.write(b'actual failed prefix')
   self.assertEqual(budget.total('terminal'),20)
 def test_14_schemas_and_resource_upper_bound(self):
  core={'input256':(1,3,256,256),'mask256':(1,1,256,256),'result256':(1,3,256,256),'raw512':(1,3,512,512),'predicted_codes':(1,256),'used_codes':(1,256),'grid_mask':(1,256),'arm_code':(),'oracle_unavailable_clean_information':()};hv=0
  for name,shape in core.items():
   dtype=np.int64 if name.endswith('codes') else np.uint8 if name in ('grid_mask','arm_code','oracle_unavailable_clean_information') else np.float32;arr=np.zeros(shape,dtype);hv+=arr.nbytes;header=io.BytesIO();np.lib.format.write_array_header_1_0(header,np.lib.format.header_data_from_array_1_0(arr));self.assertLessEqual(header.tell(),512)
  self.assertEqual(hv,4985090);parity=15208448;self.assertLess(100*hv+4*parity+104*65536,c.CONTRACT['array_bytes']);self.assertLess(c.CONTRACT['array_bytes']+c.CONTRACT['PNG_bytes_total']+c.CONTRACT['metadata_bytes']+c.CONTRACT['terminal_bytes'],c.CONTRACT['output_bytes']);self.assertNotIn("'input512'} if bridge",SOURCE)
 def test_15_actual_hash_receipt_keeps_nonfinite_bytes(self):
  import torch
  values={'encoder.'+str(i):torch.tensor([float(i)]) for i in range(497)};values['encoder.17'][0]=float('nan');fake=types.SimpleNamespace(state_dict=lambda:values,parameters=lambda:[],modules=lambda:[]);env={'torch':torch,'hashlib':hashlib,'net':fake};exec(compile(ast.Module(body=[nested('state')],type_ignores=[]),'<actual state hash helper plain tensors>','exec'),env);receipt=env['state']();self.assertEqual(receipt['state_tensors'],497);self.assertFalse(receipt['all_tensors_finite']);self.assertFalse(receipt['tensors']['encoder.17']['finite']);self.assertEqual(len(receipt['tensors']['encoder.17']['sha256']),64)
 def test_16_failure_capture_actual_toy_state_once_no_scientific_gate(self):
  import torch,random,zipfile,os
  with tempfile.TemporaryDirectory(dir=TMP) as t:
   budget=self.budget(t);values={'encoder.'+str(i):torch.tensor([float(i)]) for i in range(497)};values['encoder.17'][0]=float('nan');fake=types.SimpleNamespace(state_dict=lambda:values);env={'net':fake,'budget':budget,'out':budget.out,'need':c.need,'CONTRACT':c.CONTRACT,'PARENT_LOGICAL':c.PARENT_LOGICAL,'ORIGINAL_SHA':c.ORIGINAL_SHA,'PARENT_SHA':c.PARENT_SHA,'expected':'toyprotocol','counts':{'forward_slots':1},'typed':{'H':1},'progress':{'phase':'fixture'},'witness':{'quantizer':1},'torch':torch,'np':np,'random':random,'zipfile':zipfile,'PurePosixPath':__import__('pathlib').PurePosixPath,'os':os,'sha':lambda p:hashlib.sha256(p.read_bytes()).hexdigest(),'record':lambda path,value,emergency=False:budget.json(path,value,emergency=emergency),'failure_state_attempted':False}
   fn=copy.deepcopy(nested('retain_failed_actual_state'));fn.body=[n for n in fn.body if not isinstance(n,ast.Nonlocal)];fn.body.insert(0,ast.Global(names=['failure_state_attempted']));ast.fix_missing_locations(fn);exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual failure capture on plain toy state>','exec'),env)
   with patch.object(torch.cuda,'get_rng_state_all',return_value=[torch.zeros(16,dtype=torch.uint8)]):env['retain_failed_actual_state']({'logical_sha256':'honest-nonfinite-fixture'})
   receipt=json.loads((budget.out/'failure_state_receipt.json').read_text());self.assertTrue(receipt['complete_committed_state']);self.assertFalse(receipt['partial_retained_not_usable']);saved=torch.load(budget.out/'failed_actual_model_state.pth',map_location='cpu',weights_only=True);self.assertEqual(len(saved['state_dict']),497);self.assertTrue(torch.isnan(saved['state_dict']['encoder.17']).all());self.assertEqual(saved['optimizer_updates'],0)
   with self.assertRaises(ValueError):env['retain_failed_actual_state']({'logical_sha256':'second-attempt-refused'})
 def test_17_C_bridge_and_reused_control_gate_order(self):
  self.assertLess(SOURCE.index("record(folder/'summary.json',{'complete':True,'cases':bridges"),SOURCE.index('for arm in ARMS:'))
  self.assertIn("all(r['input256_exact']",SOURCE);self.assertIn("raw512_max_error']<=1e-6",SOURCE);self.assertIn("'all96_sources_and_masks_exact':len(control_compatibility)==96",SOURCE);self.assertIn("need(np.array_equal(predicted_np,baseline_predictions[case['id']])",SOURCE);self.assertIn('all497_state_before_after_identical',SOURCE);self.assertIn("'quality_stop_or_candidate_selection':False",SOURCE)
 def test_19_exact_additive_effects_all_endpoints_signs(self):
  rows={}
  for arm,value in [('A',4.),('C',1.),('H',2.),('V',3.)]:
   for cohort in ('monitor','challenge'):
    records=[]
    for index,kind in enumerate(c.KINDS):
     for repeat in range(2):records.append({'id':cohort+str(index)+str(repeat),'mask_kind':kind,'condition':'strict_gray','style':'GRAY',**{key:value for key in (*c.METRICS,'GRAY_raw_mean_channel_spread','GRAY_PNG_mean_channel_spread')}})
    rows[(arm,cohort)]={'records':records}
  actual=c.spatial_effects(rows);endpoint=actual['paired']['monitor']['macro']['PNG_hole_L1'];self.assertEqual(endpoint['effects'],{'H_minus_A':-2.,'V_minus_A':-1.,'C_minus_A':-3.,'C_minus_H_minus_V_plus_A':0.});self.assertEqual(endpoint['exact_signs']['C_minus_H_minus_V_plus_A'],0);self.assertEqual(len(actual['GRAY']['cases']),8);self.assertEqual(actual['GRAY']['geometry']['lower_face']['GRAY_PNG_mean_channel_spread']['effects']['C_minus_A'],-3.)
 def test_20_C_bridge_does_not_invent_historical512_layout(self):
  self.assertNotIn('old_input=app_input',SOURCE);self.assertNotIn('input512_exact_from_exact_source_and_pinned_rule',SOURCE);self.assertIn('historical C has no saved input512',SOURCE)
  self.assertIn("'input256_exact':np.array_equal",SOURCE);self.assertIn("'mask256_exact':np.array_equal",SOURCE);self.assertIn("'predicted_codes_exact':np.array_equal",SOURCE);self.assertIn("'used_codes_exact':np.array_equal",SOURCE);self.assertIn("'PNG_pixels_exact':np.array_equal",SOURCE)
 def test_21_failure_metadata_preserves_actual_path_witness(self):
  for filename in ('state_on_failure.json','state_receipt_failure.json','failure.json'):
   lines=[line for line in SOURCE.splitlines() if "record(out/'"+filename+"'" in line];self.assertEqual(len(lines),1);self.assertIn("'actual_path_witness':dict(witness)",lines[0])
 def test_18_root_design_and_manual_owned_supervisor(self):
  self.assertIn('1983f9fed9bede2078ffc6a1af0dd6ca08fe6ec532e5c3de9c4ce41d4db2c657',SOURCE);self.assertIn('98c3521e5e7e0043479862923d03cc21c9b6b7c2e5371a0ffcd292d919ca1d78',SOURCE);self.assertIn('4f6ea7134e9c8628efb7deae4ca5dc137b35dc9f1d0471c796c8aa03692e63b5',SOURCE);self.assertIn('os.getpgid(child.pid)==child.pid',SOURCE);self.assertIn('hard_cap_SIGKILL',SOURCE);self.assertIn("'external_cap_seconds':900",SOURCE);self.assertNotIn('automatic_resume_permitted\':True',SOURCE)
if __name__=='__main__':
 started=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SourceTests))
 if not result.wasSuccessful():raise SystemExit(1)
 paths=[WORKER,HERE/'contracts.py',HERE/'probe_forward.py',HERE/'optimized_budget.py',HERE/'app_delivery.py',Path(__file__)];receipt={'format':'spatial-code-intervention-v1-source-only-preparation-tests','complete':True,'tests':result.testsRun,'seconds':time.monotonic()-started,'sources':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],'local_actions':{'real_model_constructions':0,'neural_network_calls':0,'optimizer_objects_or_steps':0,'gradient_queries_or_backwards':0,'real_checkpoint_reads':0,'VM_calls':0,'plain_CPU_tensor_and_toy_serialization_only':True}}
 with (HERE/'source_preparation_tests.json').open('x',encoding='utf-8',newline='\n') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
 print(json.dumps(receipt))

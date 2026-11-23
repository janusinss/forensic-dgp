"""Source/serialization and plain CPU toy-value fixtures; no model or optimizer.
The toy uses only SimpleNamespace + fixed arithmetic callbacks, never nn.Module.
"""
from pathlib import Path
import ast,copy,contextlib,hashlib,importlib.util,io,json,runpy,sys,tempfile,time,types,unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
PACKAGE='research.completion_codeformer_same_codebook_probe_v1';package=types.ModuleType(PACKAGE);package.__path__=[str(HERE)];sys.modules[PACKAGE]=package

def load(name):
 spec=importlib.util.spec_from_file_location(PACKAGE+'.'+name,HERE/(name+'.py'));m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
c=load('contracts');b=load('optimized_budget');bridge=load('probe_forward');delivery=load('app_delivery')
WORKER=HERE/'cctv_completion_codeformer_same_codebook_probe_v1_vm.py';SOURCE=WORKER.read_text();TREE=ast.parse(SOURCE);WN=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
TMP=Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs')

def original_forward():
 t=ast.parse((ROOT/'third_party/codeformer/codeformer_arch.py').read_text());cls=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='CodeFormer');return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')

class PreparationTests(unittest.TestCase):
 def test_01_python310_syntax_no_file_digest(self):
  for p in HERE.glob('*.py'):
   t=ast.parse(p.read_text(),feature_version=(3,10));self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr=='file_digest' for n in ast.walk(t)))
 def test_02_help_scientific_imports_blocked(self):
  old=__import__
  def blocked(name,*a,**kw):
   if name.split('.')[0] in ('torch','numpy','PIL'):raise AssertionError('scientific import in help '+name)
   return old(name,*a,**kw)
  with patch('builtins.__import__',blocked),patch.object(sys,'argv',[str(WORKER),'--help']),contextlib.redirect_stdout(io.StringIO()):
   with self.assertRaises(SystemExit) as ex:runpy.run_path(str(WORKER),run_name='__main__')
  self.assertEqual(ex.exception.code,0)
 def test_03_zero_optimizer_gradient_and_teacher_calls(self):
  for n in ast.walk(WN):
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):self.assertNotIn(n.func.attr,('backward','autograd','Adam','SGD','grad'))
  self.assertNotIn('torch.optim',SOURCE);self.assertNotIn('torch.save',SOURCE);self.assertNotIn('requires_grad_(True)',SOURCE)
  self.assertEqual(c.CONTRACT['forward_slots_planned'],8+4*48);self.assertEqual(c.CONTRACT['forward_slots_max'],220);self.assertEqual(c.CONTRACT['optimizer_objects'],0)
 def test_04_exact_arms_resources_and_no_checkpoint(self):
  self.assertEqual([(a['id'],a['w'],a['oracle']) for a in c.ARMS],[('A',1,False),('B',0,False),('C',1,True),('D',0,True)])
  self.assertEqual(c.CONTRACT['output_bytes'],1536*1024**2);self.assertEqual((c.CONTRACT['worker_seconds'],c.CONTRACT['cooperative_stop_seconds']),(900,840));self.assertFalse(c.CONTRACT['checkpoint_save'])
  self.assertTrue(all('ORACLE' in a['filename_tag'] and 'unavailable_clean_information' in a['filename_tag'] for a in c.ARMS if a['oracle']))
 def test_05_bridge_AST_only_selected_index_insertion(self):
  n=copy.deepcopy(next(n for n in ast.parse((HERE/'probe_forward.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='probe_forward'))
  n.name='forward';n.args.kwonlyargs=[];n.args.kw_defaults=[]
  n.body=[v for v in n.body if not (isinstance(v,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('predicted_indices','top_idx') for t in v.targets) and ((any(isinstance(t,ast.Name) and t.id=='predicted_indices' for t in v.targets)) or (isinstance(v.value,ast.Call) and isinstance(v.value.func,ast.Name) and v.value.func.id=='select_code_indices')))]
  n.body[-1].value.elts=n.body[-1].value.elts[:3]
  self.assertEqual(ast.dump(n),ast.dump(original_forward()))
 def test_06_state_and_historical_gate_precede_oracles(self):
  self.assertIn('len(hashes)==497',SOURCE);self.assertIn('after==before',SOURCE);self.assertIn('len(prior_names)==165',SOURCE)
  self.assertLess(SOURCE.index("record(out/'ACTUAL_R1_48case_compatibility.json'"),SOURCE.index("record(out/('arm_'+arm['id']+'_complete.json')"))
  self.assertIn("if arm['id']=='A':",SOURCE);self.assertIn("'before_any_oracle':True",SOURCE);self.assertIn("r['raw256_max_error']<=1e-6 and r['PNG_bytes_exact']",SOURCE)
  self.assertIn("'all4_saved_before_gate':True",SOURCE);self.assertIn('original_call_hook=net.register_forward_pre_hook(count_original_call)',SOURCE)
 def toy(self):
  import torch
  import torch.nn.functional as F
  trace=[];size=types.SimpleNamespace
  def enc32(x):trace.append(x.clone());return F.interpolate(x.mean(1,keepdim=True),size=(32,32),mode='bilinear',align_corners=False).repeat(1,256,1,1)
  def enc16(x):return F.interpolate(x,size=(16,16),mode='nearest')
  def embed(x):return x.mean(-1,keepdim=True).repeat(1,1,512)
  def logits(x):
   a=torch.arange(512,dtype=x.dtype)[None,None,:].repeat(x.shape[0],x.shape[1],1);return a+(x.mean(-1,keepdim=True)*.001)
  def quant(indices,shape):return indices.squeeze(-1).reshape(1,1,16,16).to(torch.float32).repeat(1,256,1,1)/512
  toy=size(connect_list=['32'],fuse_encoder_block={'32':0},encoder=size(blocks=[enc32,enc16]),position_emb=torch.zeros((256,512)),feat_emb=embed,ft_layers=[],idx_pred_layer=logits,quantize=size(get_codebook_feat=quant),fuse_generator_block={'32':0},generator=size(blocks=[lambda x:F.interpolate(x,size=(32,32),mode='nearest')]),fuse_convs_dict={'32':lambda enc,dec,w:dec+enc*w})
  node=original_forward();env={'torch':torch,'F':F,'adaptive_instance_normalization':lambda a,b:a};exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned pure toy forward>','exec'),env)
  return torch,F,toy,trace,env['forward']
 def test_07_plain_toy_ordinary_forward_matches_original(self):
  torch,F,toy,trace,original=self.toy();x=torch.linspace(-1,1,3*64*64).reshape(1,3,64,64)
  with torch.no_grad():
   for w in (0,1):
    expected=original(toy,x,w=w,detach_16=True,adain=False);actual=bridge.probe_forward(toy,x,w=w,F=F,torch=torch,adaptive_instance_normalization=lambda a,b:a)
    for a,z in zip(actual[:3],expected):self.assertTrue(torch.equal(a,z))
    self.assertTrue(torch.equal(actual[3],actual[4]))
  self.assertFalse(isinstance(toy,torch.nn.Module))
 def test_08_oracle_only_indices_change_observed_context_stays(self):
  torch,F,toy,trace,original=self.toy();x=torch.linspace(-1,1,3*64*64).reshape(1,3,64,64);oracle=torch.zeros((1,256,1),dtype=torch.int64)
  with torch.no_grad():
   actual=bridge.probe_forward(toy,x,w=1,F=F,torch=torch,adaptive_instance_normalization=lambda a,b:a)
   changed=bridge.probe_forward(toy,x,w=1,F=F,torch=torch,adaptive_instance_normalization=lambda a,b:a,oracle_indices=oracle)
  self.assertTrue(torch.equal(trace[0],trace[1]));self.assertTrue(torch.equal(actual[1],changed[1]));self.assertTrue(torch.equal(actual[2],changed[2]));self.assertTrue(torch.equal(actual[3],changed[3]));self.assertTrue(torch.equal(changed[4],oracle));self.assertFalse(torch.equal(actual[0],changed[0]))
 def test_09_override_shape_range_dtype_alias_refused(self):
  import torch
  a=torch.ones((1,256,1),dtype=torch.int64);v=torch.zeros_like(a);z=bridge.select_code_indices(a,v,codebook_size=512);v.fill_(3);self.assertTrue(torch.equal(z,torch.zeros_like(a)));self.assertIs(bridge.select_code_indices(a,None,codebook_size=512),a)
  for bad in (a.to(torch.float32),a.squeeze(-1),a*512,a*-1):
   with self.assertRaises(ValueError):bridge.select_code_indices(a,bad,codebook_size=512)
 def budget(self,text,clock=lambda:0):
  out=Path(text)/'outputs';out.mkdir();return b.OutputBudget(out,0,clock=clock,disk_free=lambda:100*1024**3)
 def test_10_native_npz_png_exact_callbacks(self):
  with tempfile.TemporaryDirectory(dir=TMP) as t:
   budget=self.budget(t);p=budget.out/'toy.npz';a=np.zeros((1,3,256,256),np.float32)
   with budget.open(p) as f:np.savez_compressed(f,result256=a)
   with np.load(p,allow_pickle=False) as z:self.assertTrue(np.array_equal(z['result256'],a))
   q=budget.out/'toy.png'
   with budget.open(q) as f:Image.fromarray(np.zeros((256,256,3),np.uint8)).save(f,format='PNG')
   self.assertEqual(budget.total(),p.stat().st_size+q.stat().st_size);budget.assert_ledger_consistent()
 def test_11_deadline_caps_and_failed_prefix_preserved(self):
  with tempfile.TemporaryDirectory(dir=TMP) as t:
   now=[0.];budget=self.budget(t,lambda:now[0]);p=budget.out/'partial.npz'
   with budget.open(p) as f:
    f.write(b'keep');now[0]=840
    with self.assertRaises(ValueError):f.write(b'bad')
   self.assertEqual(p.read_bytes(),b'keep');budget.guard(emergency=True);now[0]=900
   with self.assertRaises(ValueError):budget.guard(emergency=True)
 def test_12_checkpoint_overwrite_traversal_path_member_refused(self):
  with tempfile.TemporaryDirectory(dir=TMP) as t:
   budget=self.budget(t)
   for name in ('checkpoint.pth','foo.pt.partial'):
    with self.assertRaises(ValueError):budget.open(budget.out/name)
   p=budget.out/'ok.json'
   with budget.open(p) as f:f.write(b'{}')
   with self.assertRaises(ValueError):budget.open(p)
   with self.assertRaises(ValueError):budget.open(budget.out/'..'/'outside.npz')
   with self.assertRaises(ValueError):budget.open(budget.out/('x'*240+'.npz'))
   for i in range(994):budget.entries[budget.out/str(i)]=None
   with self.assertRaises(ValueError):budget.open(budget.out/'full.npz')
 def test_13_exact_NPZ_schema_header_and_conservative_bound(self):
  shapes={'input256':(1,3,256,256),'mask256':(1,1,256,256),'result256':(1,3,256,256),'raw512':(1,3,512,512),'predicted_codes':(1,256),'used_codes':(1,256),'arm_code':(),'oracle_unavailable_clean_information':()}
  total=0
  for name,shape in shapes.items():
   dtype=np.int64 if name.endswith('codes') else np.uint8 if not shape else np.float32;a=np.zeros(shape,dtype);h=io.BytesIO();np.lib.format.write_array_header_1_0(h,np.lib.format.header_data_from_array_1_0(a));self.assertLessEqual(h.tell(),512);total+=a.nbytes
  self.assertEqual(total,4984834);self.assertLess(192*total+60833792+192*256*1024+8*1024**2+4*1024**2,c.CONTRACT['output_bytes'])
  self.assertIn('saved_npz_count<196',SOURCE);self.assertIn('saved_npz_count==196',SOURCE)
 def test_14_supervisor_finite_owned_shutdown_preservation(self):
  self.assertIn('os.getpgid(child.pid)==child.pid',SOURCE);self.assertIn('stdout_bytecap_SIGTERM',SOURCE);self.assertIn('hard_cap_SIGKILL',SOURCE);self.assertIn("'external_cap_seconds':900",SOURCE)
  self.assertIn("'partial_outputs_retained':True",SOURCE);self.assertIn("'no_learned_checkpoint_expected':True",SOURCE)
 def test_15_frozen_prior_source_hash_and_design_binding(self):
  self.assertEqual(hashlib.sha256((ROOT/'third_party/codeformer/codeformer_arch.py').read_bytes()).hexdigest(),'ee9758522173b72fc1f292df825b9a32179aebfd96142ececf9aa68f6e12e9eb')
  self.assertIn('2ca6fafbaf2a3008459dace2d83bfc0cecb6594ae6e33adc7ac4aaef5bd0effb',SOURCE);self.assertIn('4c9b2a1da70e508e0482f06235f079ab513dc3910c99073c7c678a67886cef86',SOURCE)

if __name__=='__main__':
 start=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PreparationTests))
 if not result.wasSuccessful():raise SystemExit(1)
 paths=[WORKER,HERE/'probe_forward.py',HERE/'contracts.py',HERE/'optimized_budget.py',HERE/'app_delivery.py',Path(__file__)]
 receipt={'format':'same-codebook-probe-v1-source-only-preparation-tests','complete':True,'tests':result.testsRun,'seconds':time.monotonic()-start,'sources':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],'local_actions':{'real_model_constructions':0,'neural_network_calls':0,'optimizer_objects_or_steps':0,'gradient_queries_or_backwards':0,'real_checkpoint_reads':0,'VM_calls':0,'plain_CPU_toy_arithmetic_only':True}}
 p=HERE/'source_preparation_tests.json'
 if p.exists():raise FileExistsError('Preserve earlier receipt')
 p.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))

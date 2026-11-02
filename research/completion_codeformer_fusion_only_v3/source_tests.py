"""V3 pure contracts and tiny native serialization; no neural/autograd/optimizer/VM calls."""
from pathlib import Path
import ast,contextlib,hashlib,io,json,os,random,runpy,shutil,sys,tempfile,time,types,unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.completion_codeformer_fusion_only_v3 import contracts as c
from research.completion_codeformer_fusion_only_v3.optimized_budget import OutputBudget
WORKER=ROOT/'scripts/cctv_completion_codeformer_fusion_only_v3_vm.py'
TMP=Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs')
SOURCE=WORKER.read_text(encoding='utf-8');TREE=ast.parse(SOURCE);WORKER_NODE=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='worker')

def extracted(names,env):
 nodes=[n for n in WORKER_NODE.body if isinstance(n,ast.FunctionDef) and n.name in names]
 if len(nodes)!=len(names):raise AssertionError('Missing helper')
 exec(compile(ast.Module(body=nodes,type_ignores=[]),'<extracted source helpers>','exec'),env)
 return env

class PureSourceTests(unittest.TestCase):
 def budget(self,text,clock=None):
  out=Path(text)/'out';out.mkdir();return OutputBudget(out,0,clock=clock or (lambda:0),disk_free=lambda:100*1024**3)
 def test_01_help_no_torch_import(self):
  original=__import__
  def blocked(name,*a,**kw):
   if name=='torch' or name.startswith('torch.'):raise AssertionError('Torch imported in pure help')
   return original(name,*a,**kw)
  with patch('builtins.__import__',blocked),patch.object(sys,'argv',[str(WORKER),'--help']),contextlib.redirect_stdout(io.StringIO()):
   with self.assertRaises(SystemExit) as ex:runpy.run_path(str(WORKER),run_name='__main__')
  self.assertEqual(ex.exception.code,0)
 def test_02_python310_source_and_exact_work_counts(self):
  for p in [WORKER,Path(c.__file__),ROOT/'research/completion_codeformer_fusion_only_v3/optimized_budget.py']:
   tree=ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,10));self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr=='file_digest' for n in ast.walk(tree)))
  self.assertEqual((c.CONTRACT['forward_slots_planned'],c.CONTRACT['forward_slots_max'],c.CONTRACT['backward_calls_max'],c.CONTRACT['optimizer_steps_max']),(480,600,256,64))
  self.assertEqual(8+256+128+32+56,480);self.assertEqual(c.CONTRACT['teacher_component_calls_planned'],0)
 def test_03_no_code_route_updates_or_teacher_calls(self):
  calls=[n for n in ast.walk(WORKER_NODE) if isinstance(n,ast.Call)]
  self.assertFalse(any(isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='code_optimizer' and n.func.attr in ('step','zero_grad') for n in calls))
  self.assertFalse(any(isinstance(n.func,ast.Name) and n.func.id=='forward' and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value in ('fit_code','teacher_encoder','teacher_quantizer') for n in calls))
  self.assertEqual(sum(isinstance(n.func,ast.Attribute) and n.func.attr=='backward' for n in calls),1)
 def test_04_historical48gate_precedes_training_loop(self):
  start=SOURCE.index("need(len(compatibility_rows)==48")
  self.assertLess(start,SOURCE.index('for cycle,case_ids in enumerate(schedule,1)'))
  self.assertIn("r['max_raw256_error']<=1e-6 and r['PNG_bytes_exact']",SOURCE)
  self.assertIn("allow_pickle=False",SOURCE);self.assertIn("'frozen443_sha256'",SOURCE)
 def test_05_gate_rejects_subgroup_tradeoff(self):
  base={'macro':{k:1. for k in c.METRICS},'geometry':{g:{k:1. for k in c.METRICS} for g in c.KINDS},'condition':{g:{k:1. for k in c.METRICS} for g in ('GRAY','RGB')}}
  good=json.loads(json.dumps(base))
  for group in [good['macro'],*good['geometry'].values(),*good['condition'].values()]:
   for k in group:group[k]=.98
  self.assertTrue(c.compare({'metrics':good},{'metrics':base},conditions=True)['prospective_numeric_screen_pass'])
  good['geometry']['cheek_object']['PNG_hole_L1']=1.001
  self.assertFalse(c.compare({'metrics':good},{'metrics':base},conditions=True)['prospective_numeric_screen_pass'])
  good['geometry']['cheek_object']['PNG_hole_L1']=1.11
  self.assertTrue(c.compare({'metrics':good},{'metrics':base},conditions=True)['catastrophic_regression_stop'])
 def test_06_npz_png_native_callback_roundtrip(self):
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   b=self.budget(text);p=b.out/'toy.npz';a=np.arange(768,dtype=np.float32).reshape(1,3,16,16)
   with b.open(p) as stream:np.savez_compressed(stream,result256=a)
   with np.load(p,allow_pickle=False) as z:self.assertTrue(np.array_equal(z['result256'],a))
   self.assertEqual(b.total('array'),p.stat().st_size)
   p=b.out/'toy.png';image=Image.fromarray(np.zeros((256,256,3),np.uint8))
   with b.open(p) as stream:image.save(stream,format='PNG');self.assertFalse(hasattr(stream,'fileno'))
   self.assertEqual(b.total('PNG'),p.stat().st_size);b.assert_ledger_consistent()
 def test_07_seek_extent_and_exclusive_overwrite(self):
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   b=self.budget(text);p=b.out/'toy.npz'
   with b.open(p) as stream:stream.write(b'0123456789');stream.seek(3);stream.write(b'ABC')
   self.assertEqual(p.read_bytes(),b'012ABC6789');self.assertEqual(b.total(),10)
   with self.assertRaises(ValueError):b.open(p)
   self.assertEqual(b.total(),10)
 def test_08_caps_preserve_failed_prefix(self):
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   b=self.budget(text);p=b.out/'terminal_training_state.pth.partial'
   with patch.dict(c.CONTRACT,{'terminal_bytes':16}):
    with b.open(p,terminal=True) as stream:
     stream.write(b'kept')
     with self.assertRaises(ValueError):stream.write(b'x'*17)
   self.assertEqual(p.read_bytes(),b'kept');self.assertFalse((b.out/'terminal_training_state.pth').exists())
 def test_09_deadline_disk_and_postcommit_reservation(self):
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   clock=[0.];b=self.budget(text,lambda:clock[0]);clock[0]=1680
   with self.assertRaises(ValueError):b.guard()
   b.guard(terminal=True);clock[0]=1800
   with self.assertRaises(ValueError):b.guard(terminal=True)
   clock[0]=0;b.disk_free=lambda:c.CONTRACT['stop_free_bytes']+1
   with b.open(b.out/'terminal_training_state.pth.partial',terminal=True) as stream:
    with self.assertRaises(ValueError):stream.write(b'AB')
 def test_10_native_torch_small_serialization(self):
  import torch
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   b=self.budget(text);p=b.out/'terminal_training_state.pth.partial';a=torch.arange(128,dtype=torch.float32)
   with b.open(p,terminal=True) as stream:torch.save({'fixture':a},stream)
   self.assertEqual(b.total('terminal'),p.stat().st_size);self.assertTrue(torch.equal(torch.load(p,weights_only=True)['fixture'],a))
 def fake_save_environment(self,text):
  import torch
  out=Path(text)/'out';out.mkdir();budget=OutputBudget(out,time.monotonic(),disk_free=lambda:100*1024**3)
  state={'encoder.fixture':torch.arange(4,dtype=torch.float32),'generator.fixture':torch.ones(4),'quantize.fixture':torch.zeros(4),'fuse_convs_dict.32.fixture':torch.arange(4,dtype=torch.float32)}
  net=types.SimpleNamespace(state_dict=lambda:state,named_parameters=lambda:[])
  opt={'state':{0:{'step':torch.tensor(32.),'exp_avg':torch.ones(4),'exp_avg_sq':torch.ones(4)}},'param_groups':[{'params':[0],'lr':1e-5}]}
  env={'torch':torch,'hashlib':hashlib,'net':net,'need':c.need,'json':json,'time':time,'started':time.monotonic(),'out':out,'write':lambda p,v,emergency=False:budget.json(p,v,emergency=emergency),'phase_events':[],'np':np,'random':random,'os':os,'shutil':shutil,'root':ROOT,'CONTRACT':c.CONTRACT,'budget':budget,'PARENT_SHA':c.PARENT_SHA,'PARENT_PROTOCOL_SHA':c.PARENT_PROTOCOL_SHA,'PARENT_LOGICAL':c.PARENT_LOGICAL,'WEIGHT_SHA':c.ORIGINAL_SHA,'parent_optimizer_restored':True,'parent_RNG_restored':True,'parent_metadata':{},'terminal_cycle':0,'counts':{'backward_calls':0,'code_optimizer_steps':0,'fusion_optimizer_steps':0,'forward_slots':0,'empty_bypasses':0},'model_forwards':{'calls':0},'expected':'toy-not-real-protocol','progress':{'phase':'pure_save_fixture'},'optimizer_parameter_names':{'code':['encoder.fixture'],'fusion':['fuse_convs_dict.32.fixture']},'schedule':[['toy']],'code_optimizer':types.SimpleNamespace(state_dict=lambda:opt),'fusion_optimizer':types.SimpleNamespace(state_dict=lambda:opt),'sha':c.sha,'guard':lambda terminal_save=False:budget.guard(terminal=terminal_save)}
  extracted(['digest_state','tree_digest','frozen_digest','phase_start','phase_end','cpu_copy','tensor_bytes','save_state'],env)
  env['frozen_state']=env['digest_state'](('quantize.','generator.'));env['frozen_all_state']=env['frozen_digest']();env['code_optimizer_baseline']=env['tree_digest'](opt)
  return env,state
 def test_11_actual_save_helper_retains_normal_and_timing(self):
  import torch
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   env,state=self.fake_save_environment(text)
   with patch.object(torch.cuda,'get_rng_state_all',return_value=[torch.arange(8,dtype=torch.uint8)]):receipt=env['save_state'](0,'pure_fixture')
   path=env['out']/'terminal_training_state.pth';saved=torch.load(path,weights_only=True)
   self.assertEqual(saved['retained_failure_gates'],[]);self.assertEqual(len(saved['trainable_state']),2);self.assertEqual(saved['failed_frozen_prior_state'],{})
   self.assertTrue((env['out']/'pre_save_receipt.json').is_file());self.assertGreater(receipt['bytes'],0);self.assertFalse((env['out']/'terminal_training_state.pth.partial').exists())
   self.assertTrue(any(e['phase']=='terminal_serialization' and e['write_callbacks']>0 for e in receipt['phase_timings']))
 def test_12_actual_save_helper_retains_nonfinite_frozen_failure(self):
  import torch
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   env,state=self.fake_save_environment(text);state['generator.fixture'][0]=float('nan');state['encoder.fixture'][0]=float('nan')
   with patch.object(torch.cuda,'get_rng_state_all',return_value=[torch.arange(8,dtype=torch.uint8)]):env['save_state'](0,'pure_expected_failure')
   saved=torch.load(env['out']/'terminal_training_state.pth',weights_only=True)
   self.assertIn('nonfinite current model state',saved['retained_failure_gates']);self.assertIn('443frozen state changed',saved['retained_failure_gates']);self.assertIn('generator.fixture',saved['failed_frozen_prior_state']);self.assertTrue(torch.isnan(saved['trainable_state']['encoder.fixture'][0]));self.assertFalse(saved['state_is_qualified'])
 def test_13_large_namespace_writes_do_not_resolve_or_scan(self):
  from research.completion_codeformer_fusion_only_v3.optimized_budget import Entry
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   b=self.budget(text)
   for i in range(620):
    p=b.out/('fixture_%04d.npz'%i);e=Entry(p,'array',1);b.entries[p]=e;b.sizes[p]=1;b.totals['array']+=1;b.extent_total+=1
   p=b.out/'terminal_training_state.pth.partial'
   with b.open(p,terminal=True) as stream:
    prior=b.check_counts['canonical_paths']
    with patch.object(Path,'resolve',side_effect=AssertionError('Per-write resolve forbidden')):
     for _ in range(20):stream.write(b'a')
   self.assertEqual(b.check_counts['canonical_paths'],prior);self.assertEqual(b.total('terminal'),20);b.assert_ledger_consistent()
 def test_14_frozen_sources_infrastructure_not_altered(self):
  pins={'scripts/cctv_completion_codeformer_coverage_v2_vm.py':'007f9504cd3cf16cc24a5b5f0f3fd1a1e4325f741bbfd374f6c544a3a6bcefda','research/completion_codeformer_coverage_v2/contracts.py':'1822d73ebe57b922cd9e79817ba5c961582a93e65935ef442f60ae07576ec442'}
  for n,h in pins.items():self.assertEqual(hashlib.sha256((ROOT/n).read_bytes()).hexdigest(),h)

if __name__=='__main__':
 start=time.monotonic();suite=unittest.defaultTestLoader.loadTestsFromTestCase(PureSourceTests);result=unittest.TextTestRunner(verbosity=2).run(suite)
 if not result.wasSuccessful():raise SystemExit(1)
 receipt={'format':'fusion-only-v3-source-preparation-tests-v1','complete':True,'tests':result.testsRun,'seconds':time.monotonic()-start,'sources':[{ 'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [WORKER,Path(c.__file__),ROOT/'research/completion_codeformer_fusion_only_v3/optimized_budget.py',Path(__file__)]],'local_actions':{'neural_calls':0,'gradients':0,'optimizer_objects_or_steps':0,'VM_calls':0,'real_model_checkpoint_reads':0,'toy_CPU_tensor_serialization_only':True}}
 path=Path(__file__).parent/'source_preparation_tests.json'
 if path.exists():raise FileExistsError('Preserve previous receipt')
 path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))

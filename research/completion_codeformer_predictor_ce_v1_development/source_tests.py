"""Source preparation tests only. No model, Torch, optimizer, gradients or VM."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime,timezone
import ast,hashlib,importlib.util,json,math,sys,tempfile,types,unittest
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
contracts=load('research.completion_codeformer_predictor_ce_v1.contracts',HERE/'contracts.py')
topology=load('predictor_test_optimizer_topology',HERE/'optimizer_topology.py')
# Establish no executable parent package imports: owned helper import name only.
for name in ('research','research.completion_codeformer_predictor_ce_v1'):
 if name not in sys.modules:module=types.ModuleType(name);module.__path__=[];sys.modules[name]=module
budget_module=load('predictor_test_budget',HERE/'optimized_budget.py')
class Tests(unittest.TestCase):
 def source(self):return (HERE/'cctv_completion_codeformer_predictor_ce_v1_vm.py').read_text()
 def test_python310_all_sources(self):
  for p in HERE.glob('*.py'):ast.parse(p.read_text(),feature_version=(3,10))
 def test_exact_formal_design_binding_guard(self):
  source=self.source();tree=ast.parse(source);verify=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify');calls=[n for n in ast.walk(verify) if isinstance(n,ast.Call)];self.assertIn('077ed97ec5dc4ca75f6b6bdc3caf8def09344113c3b7ce927d13b7b68a4f2b78',source);self.assertEqual(hashlib.sha256((ROOT/'research/completion_codeformer_predictor_ce_design_v1/root_design.json').read_bytes()).hexdigest(),'077ed97ec5dc4ca75f6b6bdc3caf8def09344113c3b7ce927d13b7b68a4f2b78');self.assertTrue(any(isinstance(v.func,ast.Name) and v.func.id=='need' for v in calls))
 def test_only_one_optimizer_constructor(self):
  tree=ast.parse(self.source());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Adam'];self.assertEqual(len(calls),1)
 def test_no_teacher_or_oracle_forward(self):
  source=self.source();self.assertNotIn('oracle_indices=',source);self.assertNotIn('hq_vqgan',source);self.assertNotIn('latent_loss=',source);self.assertNotIn('F.mse_loss(',source)
 def test_exact_metadata_partition(self):
  state=json.loads((ROOT/'outputs/completion_codeformer_same_codebook_probe_v1_return/cctv_completion_codeformer_same_codebook_probe_v1_results/outputs/state_before.json').read_text())['tensors'];selected=[n for n in state if contracts.predictor_name(n)];self.assertEqual(len(selected),114);self.assertEqual(len(state)-len(selected),383);self.assertEqual(sum(math.prod(state[n]['shape']) for n in selected),19450880)
 def test_balanced_fixed128_schedule(self):
  p=ROOT/'research/completion_codeformer_predictor_ce_design_independent_review_v1/balanced128_subset.json';self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),contracts.SUBSET_SHA);data=json.loads(p.read_text());rows=data['cases_in_reference_role_order'];flat=sum(data['schedule'],[]);self.assertEqual(len(rows),128);self.assertEqual(len(flat),len(set(flat)));self.assertEqual(len(data['schedule']),32);self.assertEqual({r['id'] for r in rows},set(flat));self.assertEqual(len({r['reference_id'] for r in rows}),32)
 def parent(self):return {'state':{77:{'step':32,'exp_avg':[1.],'exp_avg_sq':[2.]},8:{'step':32,'exp_avg':[3.],'exp_avg_sq':[4.]},99:{'step':32,'exp_avg':[5.],'exp_avg_sq':[6.]}},'param_groups':[{'params':[77,8,99],'lr':1e-5,'betas':(.9,.99),'eps':1e-8,'weight_decay':0}]}
 def test_projection_sparse_ids_and_reordered_active_names(self):
  parent=self.parent();old=deepcopy(parent);active=topology.project(parent,['encoder.a','ft.a','ft.b'],['ft.b','ft.a'],clone=deepcopy);self.assertEqual(active['state'][0],parent['state'][99]);self.assertEqual(active['state'][1],parent['state'][8]);self.assertEqual(active['param_groups'][0]['params'],[0,1]);self.assertEqual(parent,old);active['state'][0]['exp_avg'][0]=999.;self.assertEqual(parent,old)
 def test_merge_mixedsteps_preserves_inactive_and_parent(self):
  parent=self.parent();old=deepcopy(parent);active=topology.project(parent,['encoder.a','ft.a','ft.b'],['ft.b','ft.a'],clone=deepcopy)
  for v in active['state'].values():v['step']=64
  merged=topology.merge(parent,['encoder.a','ft.a','ft.b'],active,['ft.b','ft.a'],clone=deepcopy);self.assertEqual(merged['state'][77],old['state'][77]);self.assertEqual(merged['state'][8]['step'],64);self.assertEqual(merged['state'][99]['step'],64);self.assertEqual(merged['param_groups'],parent['param_groups']);self.assertEqual(parent,old);self.assertEqual(set(merged),{'state','param_groups'})
 def test_bad_optimizer_binding_refused(self):
  for names in (['encoder.a','ft.a','ft.a'],['encoder.a']):
   with self.assertRaises(ValueError):topology.ordered_ids(self.parent(),names)
  with self.assertRaises(ValueError):topology.project(self.parent(),['encoder.a','ft.a','ft.b'],['missing'],clone=deepcopy)
 def test_CE_meanall256_divisor_once_numpy_fixture(self):
  rng=np.random.default_rng(29);four=[]
  for _ in range(4):
   logits=rng.normal(size=(1,256,512)).astype(np.float32);labels=rng.integers(0,512,(1,256));l=logits.astype(np.float64);l-=l.max(-1,keepdims=True);ce=np.mean(np.log(np.exp(l).sum(-1))-np.take_along_axis(l,labels[...,None],axis=-1)[...,0]);four.append(ce)
  self.assertAlmostEqual(sum(v/4 for v in four),float(np.mean(four)),14);self.assertNotAlmostEqual(sum(v/16 for v in four),float(np.mean(four)),6)
 def gray(self):return {'records':[{'id':str(i),'style':'GRAY','mask_kind':contracts.KINDS[i%4],'GRAY_raw_mean_channel_spread':.02,'GRAY_PNG_mean_channel_spread':5.} for i in range(8)]}
 def test_gray_gate_all8macro4geometry(self):
  baseline=self.gray();now=deepcopy(baseline);self.assertTrue(contracts.compare_gray(now,baseline)['all_GRAY_macro_geometry_chroma_nonworse']);now['records'][0]['GRAY_PNG_mean_channel_spread']=6.;r=contracts.compare_gray(now,baseline);self.assertFalse(r['all_GRAY_macro_geometry_chroma_nonworse']);self.assertEqual({v['group'] for v in r['nonworse_failures']},{'macro','geometry/lower_face'})
 def test_gray_gate_missingcase_refused(self):
  now=self.gray();now['records'].pop()
  with self.assertRaises(ValueError):contracts.gray_style(now)
 def test_planned_counts(self):
  c=contracts.CONTRACT;self.assertEqual(sum(contracts.TYPED_CAPS.values()),384);self.assertEqual(8+192+56,256);self.assertEqual(32*4,128);self.assertEqual(192+64,256);self.assertEqual(256+4+128,388);self.assertEqual(4*40+8,168);self.assertEqual(c['cooperative_stop_seconds']+c['save_reserve_seconds'],c['worker_seconds'])
 def test_budget_exact_ownedextents_seek_and_overwrite_refusal(self):
  with tempfile.TemporaryDirectory() as temp:
   out=Path(temp);b=budget_module.OutputBudget(out,0,disk_free=lambda:20*1024**3,clock=lambda:1);p=out/'FIT_logits/a.npz'
   with b.open(p) as stream:stream.write(b'abcd');stream.seek(1);stream.write(b'X')
   self.assertEqual(b.total(),4);self.assertTrue(b.assert_ledger_consistent())
   with self.assertRaises(ValueError):b.open(p)
   with self.assertRaises(ValueError):b.open(out.parent/'escape.npz')
 def test_budget_terminalcap_before_write(self):
  with tempfile.TemporaryDirectory() as temp:
   out=Path(temp);b=budget_module.OutputBudget(out,0,disk_free=lambda:20*1024**3,clock=lambda:1)
   with b.open(out/'terminal_training_state.pth.partial',terminal=True) as stream:
    with self.assertRaises(ValueError):b.before_entry_write(stream.entry,contracts.CONTRACT['terminal_bytes']+1,terminal=True)
 def test_budget_workdeadline_and_savewindow(self):
  with tempfile.TemporaryDirectory() as temp:
   out=Path(temp);b=budget_module.OutputBudget(out,0,disk_free=lambda:20*1024**3,clock=lambda:1681)
   with self.assertRaises(ValueError):b.guard()
   b.guard(terminal=True)
 def test_forward_pre_dispatch_caps_and_actual_hook_fixture(self):
  tree=ast.parse(self.source());worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker');forward=next(n for n in worker.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
  class Budget:terminal_committed=False
  class Clock:
   @staticmethod
   def monotonic():return 1
  scope={'need':contracts.need,'budget':Budget(),'time':Clock,'started':0,'CONTRACT':contracts.CONTRACT,'counts':{'forward_slots':0},'typed':{k:0 for k in contracts.TYPED_CAPS},'TYPED_CAPS':contracts.TYPED_CAPS,'guard':lambda:None,'model_forwards':{'calls':0}}
  exec(compile(ast.fix_missing_locations(ast.Module(body=[forward],type_ignores=[])),'<exact-forward>','exec'),scope)
  calls=[]
  def invoke():calls.append(1);scope['model_forwards']['calls']+=1;return 'OK'
  self.assertEqual(scope['forward']('fit_code',invoke),'OK');self.assertEqual(len(calls),1)
  scope['typed']['fit_code']=128
  with self.assertRaises(ValueError):scope['forward']('fit_code',invoke)
  self.assertEqual(len(calls),1)
  scope['typed']['fit_code']=0;scope['counts']['forward_slots']=420;scope['model_forwards']['calls']=420
  with self.assertRaises(ValueError):scope['forward']('fit_code',invoke)
  self.assertEqual(len(calls),1);self.assertEqual(scope['counts']['forward_slots'],420)
 def test_single_terminal_attempt_is_pre_assembly(self):
  tree=ast.parse(self.source());worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker');save=next(n for n in worker.body if isinstance(n,ast.FunctionDef) and n.name=='save_state');body=ast.get_source_segment(self.source(),save);self.assertLess(body.index("need(not terminal_attempted"),body.index("merged=merge"));self.assertLess(body.index("terminal_attempted=True"),body.index("actual_state_at_save.json"))
 def test_terminal_internal_caps_and_namespace(self):
  import zipfile
  terminal=load('preparation_terminal_admission',HERE/'terminal_admission.py')
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp)
   def pack(name,extra=(),pickle_size=1):
    p=root/name
    with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_STORED) as z:
     z.writestr('archive/data.pkl',b'x'*pickle_size);z.writestr('archive/data/0',b'1234');z.writestr('archive/version',b'3')
     for item in extra:z.writestr(item,b'x')
    return p
   self.assertTrue(terminal.admit_terminal_zip(pack('ok.partial'),guard=lambda:None)['all_member_CRC_pass'])
   for path in (pack('unsafe.partial',['archive/../escape']),pack('script.partial',['archive/run.py']),pack('pickle.partial',pickle_size=1024**2+1),pack('entries.partial',['archive/data/'+str(i) for i in range(1,1615)])):
    with self.assertRaises(ValueError):terminal.admit_terminal_zip(path,guard=lambda:None)
    self.assertTrue(path.exists())
 def test_budget_path_and_member_reservations(self):
  with tempfile.TemporaryDirectory() as temp:
   out=Path(temp);b=budget_module.OutputBudget(out,0,disk_free=lambda:20*1024**3,clock=lambda:1)
   with self.assertRaises(ValueError):b.open(out/('a'*240+'.npz'))
   b.entries={str(i):None for i in range(996)}
   with self.assertRaises(ValueError):b.open(out/'extra.json')
 def test_NPZ_prewrite_header_and_name_limits(self):
  source=self.source();tree=ast.parse(source);worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker');save=next(n for n in worker.body if isinstance(n,ast.FunctionDef) and n.name=='save_npz');body=ast.get_source_segment(source,save);self.assertLess(body.index('header.tell()<=512'),body.index('budget.open(path)'));self.assertLess(body.index('<=80'),body.index('budget.open(path)'));self.assertIn('saved_npz_count<388',body)
 def test_terminal_guard_retains_failed_scientific_state(self):
  from types import SimpleNamespace
  tree=ast.parse(self.source());worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker');guard=next(n for n in worker.body if isinstance(n,ast.FunctionDef) and n.name=='guard')
  calls=[]
  class Budget:
   def guard(self,*,terminal=False):calls.append(terminal)
  class Net:
   def modules(self):return [SimpleNamespace(training=True)]
   def named_parameters(self):return [('encoder.bad',SimpleNamespace(requires_grad=True,grad=object()))]
  scope={'need':contracts.need,'budget':Budget(),'counts':{'forward_slots':0,'backward_calls':0,'predictor_optimizer_steps':0,'empty_bypasses':0},'typed':{k:0 for k in contracts.TYPED_CAPS},'TYPED_CAPS':contracts.TYPED_CAPS,'torch':SimpleNamespace(cuda=SimpleNamespace(memory_allocated=lambda:0)),'CONTRACT':contracts.CONTRACT,'net':Net(),'predictor_name':contracts.predictor_name}
  exec(compile(ast.fix_missing_locations(ast.Module(body=[guard],type_ignores=[])),'<exact-guard>','exec'),scope)
  with self.assertRaises(ValueError):scope['guard']()
  scope['guard'](terminal_save=True);self.assertEqual(calls,[False,True])
  scope['counts']['forward_slots']=999;scope['guard'](terminal_save=True)
  class FailingBudget:
   def guard(self,*,terminal=False):raise ValueError('Resource failure')
  scope['budget']=FailingBudget()
  with self.assertRaises(ValueError):scope['guard'](terminal_save=True)
 def test_no_scientific_execution(self):self.assertFalse(any(n=='torch' or n.startswith('torch.') for n in sys.modules))
if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests);result=unittest.TextTestRunner(verbosity=2).run(suite);record={'format':'predictor-ce-v1-author-source-preparation-tests-v1','complete':result.wasSuccessful(),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'local_neural_calls':0,'model_objects':0,'optimizer_objects':0,'backward_calls':0,'VM_calls':0,'prepared_source_only_not_runtime_or_packet_admission':True,'UTC':datetime.now(timezone.utc).isoformat()}
 target=HERE/'source_preparation_tests.json';target.write_bytes((json.dumps(record,indent=2)+'\n').encode());raise SystemExit(0 if result.wasSuccessful() else 1)
